from flask import Flask, request, send_file, make_response

app = Flask(__name__)

import gc
import logging
import io
import struct
import json
import pickle

from tqdm import tqdm
from parselmouth.praat import call
from parselmouth import Sound
from librosa import load
from dsp import RobotFx
from datetime import datetime
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

import warnings

warnings.simplefilter("ignore", UserWarning)


def get_text(text, hps):
    text_norm = text_to_sequence(text, hps.data.text_cleaners)

    # if letter_switch:
    #     text_norm = apply_letter_switch(text_norm, letter_switch)

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

        m_p = torch.matmul(attn.squeeze(1), m_p.transpose(1, 2)).transpose(1, 2)
        logs_p = torch.matmul(attn.squeeze(1), logs_p.transpose(1, 2)).transpose(1, 2)

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


def convert_weights_to_speaker_embedding(vector, n_comp, algorithm_name='PCA'):
    scaler = pickle.load(open('embeddings/vctk-vits_fitted_standard_scaler.sav', 'rb'))
    assert algorithm_name in ['PLS_reg', 'PLS_can', 'CCA', 'PCA']
    model = pickle.load(open(f'embeddings/vctk-vits_fitted_{algorithm_name}-{n_comp}.sav', 'rb'))
    vector = np.array(vector)
    assert vector.shape[1] == n_comp
    pred = model.inverse_transform(vector)
    return scaler.inverse_transform(pred)


def normalize(audio, rms_level=-6):
    """
     Normalize the signal given a certain technique (peak or rms).
     Args:
         - audio    (np array) : waveform
         - rms_level (int) : rms level in dB.
     """
    # linear rms level and scaling factor
    r = 10 ** (rms_level / 10.0)
    a = np.sqrt((len(audio) * r ** 2) / np.sum(audio ** 2))

    # normalize
    return audio * a


@app.route('/synthesize', methods=["POST", "GET"])
def generate():
    gc.collect()
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
    assert ("weights" in data) or ("speaker_embeddings" in data)
    if "weights" in data:
        # Only specific to dimension reduction
        assert "algorithm_name" in data
        assert "number" in data

    assert "n_slider_ticks" in data
    N_SLIDER_TICKS = int(data["n_slider_ticks"])

    effect_ticks = {
        'pitch': [
            0,  # off
            0.08, 0.16,
            0.25,  # slightly on
            0.27, 0.29, 0.31,
            0.33,  # somewhat on
            0.35, 0.37, 0.40,
            0.42,  # on
            0.44, 0.46, 0.48,
            0.50,  # max
        ],
        'tremolo': [
            0,  # off
            0.07, 0.13,
            0.20,  # slightly on
            0.22, 0.24, 0.25,
            0.27,  # somewhat on
            0.29, 0.31, 0.32,
            0.34,  # on
            0.36, 0.37, 0.39,
            0.40,  # max
        ],
        'flanger': [
            0,  # off
            0.08, 0.16,
            0.25,  # slightly on
            0.29, 0.33, 0.36,
            0.40,  # somewhat on
            0.43, 0.45, 0.48,
            0.50,  # on
            0.53, 0.55, 0.57,
            0.60,  # max
        ],
        'griffin': [
            0,  # off
            0.12, 0.24,
            0.35,  # slightly on
            0.40, 0.45, 0.50,
            0.55,  # somewhat on
            0.61, 0.68, 0.75,
            0.81,  # on
            0.86, 0.91, 0.95,
            1.00,  # max
        ],
    }

    def normalize_value(effect_name, value, zero_idx=0):
        assert type(value) == int or type(value) == float
        assert value >= 0
        assert value < N_SLIDER_TICKS, f"Effect {effect_name} must be < {N_SLIDER_TICKS}"
        n = (N_SLIDER_TICKS - 1)
        return (value / n) - (zero_idx / n)

    def get_slider_value(effect_name, value, zero_idx=0):
        if effect_name in effect_ticks:
            return effect_ticks[effect_name][value]
        else:
            return normalize_value(effect_name, value, zero_idx)


    effects = {}
    for effect_name in ["pitch", "tremolo", "griffin", "flanger", "speed"]:
        assert effect_name in data, f"Effect {effect_name} not found in data"
        value = data[effect_name]
        if type(value) == str:
            value = eval(value)

        zero_idx = 0 if effect_name != "speed" else 7
        if type(value) == list:
            value = [get_slider_value(effect_name, v, zero_idx) for v in value]
        elif type(value) == int or type(value) == float:
            value = [get_slider_value(effect_name, value, zero_idx)]
        else:
            raise ValueError(f"Effect {effect_name} has invalid value {value}")
        effects[effect_name] = value
    print(effects)

    with tempfile.TemporaryDirectory() as out_dir:
        key = 'tmp.wav'
        use_weights = "weights" in data
        if use_weights:
            weights = data["weights"]
            if isinstance(weights, str):
                weights = json.loads(weights)
        text = data['text']
        output_file = os.path.join(out_dir, key)
        bname = key.split('.')[0]
        if use_weights:
            speaker_embeddings = convert_weights_to_speaker_embedding(weights, int(data['number']),
                                                                      data['algorithm_name'])
        else:
            speaker_embeddings = np.array(data["speaker_embeddings"], dtype=np.float32)

        audio_paths = []
        stn_tst = get_text(text, hps)
        for i in tqdm(range(speaker_embeddings.shape[0])):
            speaker_embedding = speaker_embeddings[i]
            fx_factors = {k: v[i] for k, v in effects.items()}

            spk_emb = torch.from_numpy(speaker_embedding.astype(np.float32))[None]

            tmp_wav_path = os.path.join(out_dir, bname + '_' + str(i) + '.wav')

            # stn_tst = get_text(text, hps, get_param(data['letter_switch'], i))

            with torch.no_grad():
                x_tst = stn_tst.unsqueeze(0)
                x_tst_lengths = torch.LongTensor([stn_tst.size(0)])
                torch.manual_seed(0)
                np.random.seed(0)
                audio = net_g.infer(
                    x_tst,
                    x_tst_lengths,
                    spk_emb=spk_emb,
                    noise_scale=.667,
                    noise_scale_w=0.8,
                    length_scale=1
                )[0][0, 0].data.cpu().float().numpy()

            sr = hps.data.sampling_rate

            additional_parameters = {
                'pitch_semitones': 5,
                'pitch_mirror': True,  # default mirror
                'griffin_iters': 0,  # highest compression
                'flanger_delay': 1,
                'flanger_depth': 10,
                'flanger_frequency': 5,
            }

            #fx_factors['pitch'] = 1

            timestretch = 1 - fx_factors['speed']

            # Fill in the defaults

            # Change duration
            if timestretch != 1:
                sound = Sound(audio, sr)
                manipulation = call(sound, "To Manipulation", 0.01, 75, 600)
                duration_tier = call("Create DurationTier", "tmp", 0, sound.xmax - sound.xmin)
                call([duration_tier], "Add point", 0, timestretch)
                call([duration_tier, manipulation], "Replace duration tier")
                sound = call(manipulation, "Get resynthesis (overlap-add)")
                call(sound, "Save as WAV file", tmp_wav_path)
                audio, sr = load(tmp_wav_path)

            del fx_factors['speed']

            fx = RobotFx(sr)
            y = fx.process_audio(audio, fx_factors, additional_parameters).astype(np.float32)

            # Generate wav
            write(tmp_wav_path, sr, normalize(y))
            audio_paths.append(tmp_wav_path)

        if len(audio_paths) == 1:
            output_file = tmp_wav_path
        else:
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
