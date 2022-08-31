from flask import Flask, request, send_file, jsonify, make_response
import logging
import io
import struct
import json
import pickle
from copy import copy

from dsp import RobotFx

from datetime import datetime

################
# Load VITS
################
from scipy.io.wavfile import write

from vits import commons
from vits import utils
from vits.models import SynthesizerTrn
from vits.text.symbols import symbols
from vits.text import text_to_sequence

import numpy as np
import os
import torch
import tempfile

def draw_random_int_with_blocklist(start, end, block_list):
    all_idxs = range(start, end)
    available_idxs = [idx for idx in all_idxs if idx not in block_list]
    return np.random.choice(available_idxs, 1)[0]


def apply_letter_switch(text_norm, param):
    num_switches = int(np.round(param))
    print(f'{num_switches} switches')
    to_idxs = []
    for i in range(num_switches):
        # Avoid moving an already switched letter
        from_idx = draw_random_int_with_blocklist(0, len(text_norm), to_idxs)

        # Avoid overwriting a switched letter
        to_idx = draw_random_int_with_blocklist(0, len(text_norm), to_idxs + [from_idx])
        to_idxs.append(to_idx)
        print(f'Switch {from_idx} to {to_idx}')
        print(len(to_idxs))
        text_norm[from_idx], text_norm[to_idx] = text_norm[to_idx], text_norm[from_idx]
    print(text_norm)
    return text_norm


def get_text(text, hps, letter_switch):
    text_norm = text_to_sequence(text, hps.data.text_cleaners)

    if letter_switch:
        text_norm = apply_letter_switch(text_norm, letter_switch)

    if hps.data.add_blank:
        text_norm = commons.intersperse(text_norm, 0)
    text_norm = torch.LongTensor(text_norm)
    return text_norm


class CustomVITS(SynthesizerTrn):
    def infer(self, x, x_lengths, sid=None, noise_scale=1, length_scale=1, noise_scale_w=1., max_len=None,
              spk_emb=None):
        assert sum([1 for s in [sid, spk_emb] if s is not None]) == 1, \
            "You can either set the sid or speaker embedding."

        x, m_p, logs_p, x_mask = self.enc_p(x, x_lengths)
        if self.n_speakers > 0:
            if sid is not None:
                g = self.emb_g(sid).unsqueeze(-1)  # [b, h, 1]
            elif spk_emb is not None:
                g = spk_emb.unsqueeze(-1)
            else:
                NotImplementedError('This may not happen')
        else:
            g = None

        if self.use_sdp:
            logw = self.dp(x, x_mask, g=g, reverse=True, noise_scale=noise_scale_w)
        else:
            logw = self.dp(x, x_mask, g=g)
        w = torch.exp(logw) * x_mask * length_scale
        w_ceil = torch.ceil(w)
        y_lengths = torch.clamp_min(torch.sum(w_ceil, [1, 2]), 1).long()
        y_mask = torch.unsqueeze(commons.sequence_mask(y_lengths, None), 1).to(x_mask.dtype)
        attn_mask = torch.unsqueeze(x_mask, 2) * torch.unsqueeze(y_mask, -1)
        attn = commons.generate_path(w_ceil, attn_mask)

        m_p = torch.matmul(attn.squeeze(1), m_p.transpose(1, 2)).transpose(1,
                                                                           2)  # [b, t', t], [b, t, d] -> [b, d, t']
        logs_p = torch.matmul(attn.squeeze(1), logs_p.transpose(1, 2)).transpose(1,
                                                                                 2)  # [b, t', t], [b, t, d] -> [b, d, t']

        z_p = m_p + torch.randn_like(m_p) * torch.exp(logs_p) * noise_scale
        z = self.flow(z_p, y_mask, g=g, reverse=True)
        o = self.dec((z * y_mask)[:, :, :max_len], g=g)
        return o, attn, y_mask, (z, z_p, m_p, logs_p)


def make_batch_file(in_files, output_path):
    with open(output_path, "wb") as output:
        for in_file in in_files:
            b = os.path.getsize(in_file)
            output.write(struct.pack("I", b))
            with open(in_file, "rb") as i:
                output.write(i.read())


def convert_pca_weights_to_speaker_embedding(pca_weights, pca_variation):
    N_PCA = 10
    scaler = pickle.load(open('vctk-vits_fitted_standard_scaler.sav', 'rb'))
    components = np.load(f'vctk-vits-{pca_variation}-components.npy', allow_pickle=True)

    # Make assertions
    assert components.shape[0] == N_PCA

    pca_weights = np.array(pca_weights, dtype=np.float32)
    spk_embeddings = []
    for i in range(pca_weights.shape[0]):
        pca_vector = pca_weights[i]
        assert pca_vector.shape[0] == N_PCA
        speaker_embedding = scaler.inverse_transform(np.dot(pca_vector, components))
        spk_embeddings.append(speaker_embedding.astype(np.float32))
    spk_embeddings = np.array(spk_embeddings)
    return spk_embeddings


app = Flask(__name__)

if __name__ != "__main__":
    gunicorn_logger = logging.getLogger("gunicorn.error")
    app.logger.handlers = gunicorn_logger.handlers
    app.logger.setLevel(gunicorn_logger.level)


@app.route('/synthesize', methods=["POST", "GET"])
def generate():
    hps = utils.get_hparams_from_file("vits/configs/vctk_base.json")
    net_g = CustomVITS(
        len(symbols),
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        n_speakers=hps.data.n_speakers,
        **hps.model)
    _ = net_g.eval()

    _ = utils.load_checkpoint("vits/pretrained_vctk.pth", net_g, None)

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
    assert "text" in data
    assert ("pca_weights" in data) or ("speaker_embeddings" in data)

    PCA_VARIATION = "no_rotation1"
    N = 16
    FX_NAMES = ["pitch", "tremolo", "griffin"]
    FX_MAX = 1
    EXTRA_FX_NAMES = ["flanger_frequency", "flanger_depth", "flanger_delay"]
    EXTRA_FX_MAX = 15

    x = np.linspace(0.1, 1, N)
    x_norm = (x - min(x)) / (max(x) - min(x))

    key = 'tmp.wav'

    use_pca_weights = "pca_weights" in data
    if use_pca_weights:
        pca_weights = data["pca_weights"]
        if isinstance(pca_weights, str):
            pca_weights = json.loads(pca_weights)
    text = data['text']

    fx_factors = {}
    additional_parameters = {}

    def load_params(key, max_val=1):
        assert key in data
        if type(data[key]) != str:
            d = data[key]
        else:
            d = json.loads(data[key])
        if type(d) == list:
            assert len(d) == len(
                pca_weights), 'For batch creation, you need either 1 param for all creations or one for each generation'
            return [x_norm[int(x)] * max_val for x in data[key]]
        else:
            return x_norm[int(data[key])] * max_val

    def get_param(value, i):
        if type(value) == list:
            return value[i]
        else:
            return value

    for fx_name in FX_NAMES:
        fx_factors[fx_name] = load_params(fx_name, FX_MAX)

    for extra_fx_name in EXTRA_FX_NAMES:
        additional_parameters[extra_fx_name] = load_params(extra_fx_name, EXTRA_FX_MAX)

    # Fill in the defaults
    additional_parameters['pitch_semitones'] = 7.0  # default 7 st
    additional_parameters['pitch_mirror'] = True  # default mirror
    additional_parameters['griffin_iters'] = 0  # highest compression
    fx_factors['flanger'] = 1.0  # Set to a constant

    data["letter_switch"] = load_params("letter_switch", EXTRA_FX_MAX)

    with tempfile.TemporaryDirectory() as out_dir:
        output_file = os.path.join(out_dir, key)
        bname = key.split('.')[0]
        if use_pca_weights:
            speaker_embeddings = convert_pca_weights_to_speaker_embedding(pca_weights, PCA_VARIATION)
        else:
            speaker_embeddings = np.array(data["speaker_embeddings"], dtype=np.float32)

        audio_paths = []
        for i in range(speaker_embeddings.shape[0]):
            speaker_embedding = speaker_embeddings[i]

            spk_emb = torch.from_numpy(speaker_embedding.astype(np.float32))[None]

            tmp_wav_path = os.path.join(out_dir, bname + '_' + str(i) + '.wav')

            stn_tst = get_text(text, hps, get_param(data['letter_switch'], i))
            with torch.no_grad():
                x_tst = stn_tst.unsqueeze(0)
                x_tst_lengths = torch.LongTensor([stn_tst.size(0)])
                torch.manual_seed(0)
                np.random.seed(0)
                audio = \
                    net_g.infer(x_tst, x_tst_lengths, spk_emb=spk_emb, noise_scale=.667, noise_scale_w=0.8,
                                length_scale=1)[
                        0][
                        0, 0].data.cpu().float().numpy()
            fx = RobotFx(hps.data.sampling_rate)

            _fx_factors = copy(fx_factors)
            for key, value in _fx_factors.items():
                _fx_factors[key] = get_param(value, i)

            _additional_parameters = copy(additional_parameters)
            for key, value in _additional_parameters.items():
                _additional_parameters[key] = get_param(value, i)

            app.logger.info({
                **_additional_parameters,
                **_fx_factors,
                'letter_switch': data['letter_switch']
            })

            y = fx.process_audio(audio, _fx_factors, _additional_parameters).astype(np.float32)

            # Generate wav
            write(tmp_wav_path, hps.data.sampling_rate, y)
            audio_paths.append(tmp_wav_path)

        if len(audio_paths) == 1:
            output_file = tmp_wav_path
        else:
            print(audio_paths)
            make_batch_file(audio_paths, output_file)
        app.logger.info(f'Elapsed time: {datetime.now() - begin_time}')
        with open(output_file, 'rb') as bites:
            response = make_response(send_file(
                io.BytesIO(bites.read()),
                attachment_filename=key,
                mimetype='audio/wav'
            ))
            response.headers.add('Access-Control-Allow-Origin', '*')
            response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
            response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')

            return response


if __name__ != "__main__":
    gunicorn_logger = logging.getLogger("gunicorn.error")
    app.logger.handlers = gunicorn_logger.handlers
    app.logger.setLevel(gunicorn_logger.level)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
