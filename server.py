import io
import os
import logging
import tempfile
import warnings

from flask import Flask, request, send_file, make_response, render_template
from synthesis import synthesize
from datetime import datetime

app = Flask(__name__)
warnings.simplefilter("ignore", UserWarning)


@app.route('/synthesize', methods=["POST", "GET"])
def generate():
    begin_time = datetime.now()
    app.logger.info(f"Receiving {request.method} request")
    app.logger.info("Synthesizing stimulus...")
    if request.method == 'GET':
        data = request.args.to_dict()
    elif request.method == 'POST':
        data = request.json
    else:
        raise NotImplementedError()

    app.logger.info("Data received...")

    with tempfile.TemporaryDirectory() as out_dir:
        output_file = os.path.join(out_dir, 'tmp.wav')
        synthesize(output_file, data)
        app.logger.info(f'Elapsed time: {datetime.now() - begin_time}')

        with open(output_file, 'rb') as bites:
            response = make_response(send_file(
                io.BytesIO(bites.read()),
                mimetype='audio/wav'
            ))
            response.headers.add('Access-Control-Allow-Origin', '*')
            response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
            response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')

            return response


@app.route('/')
def index():
    return render_template('client.html')


if __name__ != "__main__":
    gunicorn_logger = logging.getLogger("gunicorn.error")
    app.logger.handlers = gunicorn_logger.handlers
    app.logger.setLevel(gunicorn_logger.level)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
