# I/O
from pathlib import Path
import io
from scipy.io import wavfile

# Server support
from flask import Flask, Response, make_response, send_file, request, jsonify
from flask_cors import CORS

from robotvoice.postprocess import PostProcessor
from robotvoice import synth

# Prepare server app
app = Flask(__name__)
CORS(app)  # allows all origins (for dev)
p_proc = PostProcessor(Path("./VSTs"))

# @app.route("/list_voices", methods=["POST", "GET"])
# def list_voices() -> Response:
#     pass

@app.route("/list_postprocessing_effects", methods=["POST", "GET"])
def list_postprocessing_effects() -> Response:
    list_effects = p_proc.list_available_plugins()
    response = make_response(list_effects)
    response.headers.add("Access-Control-Allow-Origin", "*")
    response.headers.add("Access-Control-Allow-Headers", "Content-Type,Authorization")
    response.headers.add("Access-Control-Allow-Methods", "GET,PUT,POST,DELETE,OPTIONS")

    return response

@app.route("/list_effect_info", methods=["POST", "GET"])
def get_effect_configuration() -> Response:

    # Extract the text to synthesize
    if request.method == "GET":
        request_data = request.args.to_dict()
    elif request.method == "POST":
        request_data = request.json
    else:
        raise NotImplementedError(f"The method {request.method} is not supported")

    plugin_name = str(request_data["plugin_name"])

    list_effects = p_proc.list_info_plugin(plugin_name)
    response = make_response(list_effects)
    response.headers.add("Access-Control-Allow-Origin", "*")
    response.headers.add("Access-Control-Allow-Headers", "Content-Type,Authorization")
    response.headers.add("Access-Control-Allow-Methods", "GET,PUT,POST,DELETE,OPTIONS")

    return response

@app.route("/set_effect_info", methods=["POST", "GET"])
def apply_effect_configuration() -> Response:

    # Extract the text to synthesize
    if request.method == "GET":
        request_data = request.args.to_dict()
    elif request.method == "POST":
        request_data = request.json
    else:
        raise NotImplementedError(f"The method {request.method} is not supported")

    plugin_name = str(request_data["plugin_name"])
    plugin_conf = request_data["plugin_configuration"]
    p_proc.configure_plugin(plugin_name, plugin_conf)

    response = make_response(True)
    response.headers.add("Access-Control-Allow-Origin", "*")
    response.headers.add("Access-Control-Allow-Headers", "Content-Type,Authorization")
    response.headers.add("Access-Control-Allow-Methods", "GET,PUT,POST,DELETE,OPTIONS")

    return response


@app.route("/get_effects", methods=["POST", "GET"])
def get_effects() -> Response:

    response = make_response(p_proc.get_effects())
    response.headers.add("Access-Control-Allow-Origin", "*")
    response.headers.add("Access-Control-Allow-Headers", "Content-Type,Authorization")
    response.headers.add("Access-Control-Allow-Methods", "GET,PUT,POST,DELETE,OPTIONS")

    return response


@app.route("/synth", methods=["POST", "GET"])
def synthesize() -> Response:

    # Extract the text to synthesize
    if request.method == "GET":
        request_data = request.args.to_dict()
    elif request.method == "POST":
        request_data = request.json
    else:
        raise NotImplementedError(f"The method {request.method} is not supported")

    text_to_synth = str(request_data["text"])

    parameters = {"robot_effect": False}

    # parameters = DEFAULT_PARAMETERS.copy()
    # if ("robot_effect" in request_data) and request_data["robot_effect"]:
    #     for param in DEFAULT_PARAMETERS.keys():
    #         if param in request_data:
    #             parameters[param] = request_data[param]
    #             parameters["robot_effect"] = True

    audio, sr = synth.synthesize(text_to_synth, parameters)
    byte_io = io.BytesIO()
    wavfile.write(byte_io, sr, audio)

    response = make_response(send_file(io.BytesIO(byte_io.read()), mimetype="audio/wav"))
    response.headers.add("Access-Control-Allow-Origin", "*")
    response.headers.add("Access-Control-Allow-Headers", "Content-Type,Authorization")
    response.headers.add("Access-Control-Allow-Methods", "GET,PUT,POST,DELETE,OPTIONS")

    return response

@app.route('/api', methods = ['GET'])
def this_func():
    """This is a function. It does nothing."""
    return jsonify({ 'result': '' })

@app.route('/api/help', methods = ['GET'])
def help():
    """Print available functions."""
    func_list = {}
    for rule in app.url_map.iter_rules():
        if rule.endpoint != 'static':
            func_list[rule.rule] = app.view_functions[rule.endpoint].__doc__
    return jsonify(func_list)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5001)
