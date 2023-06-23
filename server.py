from flask import Flask, request, send_file, make_response, render_template

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
from dsp import Fx
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


SUPPORTED_VOICE_EFFECTS = ['speed', 'pitch_shift', 'pitch_range']

ROBOT_EFFECT_TICKS = {
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
    'flanger': np.array([
        0,  # off
        0.08, 0.16,
        0.25,  # slightly on
        0.29, 0.33, 0.36,
        0.40,  # somewhat on
        0.43, 0.45, 0.48,
        0.50,  # on
        0.53, 0.55, 0.57,
        0.60,  # max
    ]) * 1.3,
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
    'timeshift': [
        0,  # off
        3, 6,
        9,  # slightly on
        12, 15, 18,
        21,  # somewhat on
        24, 27, 30,
        33,  # on
        36, 39, 42,
        45,  # max
    ],
    'flanger_type': [
        1, 2, 3, 4, 5
    ],
    'vocoder': [
        0,  # off
        0.03, 0.06,
        0.08,  # slightly on
        0.10, 0.12, 0.14,
        0.15,  # somewhat on
        0.18, 0.21, 0.24,
        0.25,  # on
        0.28, 0.31, 0.34,
        0.35,  # max
    ],
    'vocoder_type': [
        1, 2, 3, 4, 5
    ],

}

SUPPORTED_ROBOT_EFFECTS = list(ROBOT_EFFECT_TICKS.keys())


def normalize_value(effect_name, value, zero_idx, n_slider_ticks):
    assert type(value) == int or type(value) == float
    assert value >= 0
    assert value < n_slider_ticks, f"Effect {effect_name} must be < {n_slider_ticks}"
    n = (n_slider_ticks - 1)
    return (value / n) - (zero_idx / n)


def get_slider_value(effect_name, value, zero_idx, n_slider_ticks):
    if effect_name in ROBOT_EFFECT_TICKS:
        return ROBOT_EFFECT_TICKS[effect_name][value]
    else:
        return normalize_value(effect_name, value, zero_idx, n_slider_ticks)


def get_effects_dict(data, supported_effects, n_slider_ticks, zero_idx=0):
    effects = {}
    for effect_name in supported_effects:
        assert effect_name in data, f"Effect {effect_name} not found in data"
        value = data[effect_name]

        if type(value) == str:
            value = eval(value)


        if type(value) == list:
            value = [get_slider_value(effect_name, v, zero_idx, n_slider_ticks) for v in value]
        elif type(value) == int or type(value) == float:
            value = [get_slider_value(effect_name, value, zero_idx, n_slider_ticks)]
        else:
            raise ValueError(f"Effect {effect_name} has invalid value {value}")

        effects[effect_name] = value
    return effects


def get_current_effect_values(effect_dict_list, index):
    return {k: v[index] for k, v in effect_dict_list.items()}


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
    n_slider_ticks = int(data["n_slider_ticks"])

    voice_effects_dict_list = get_effects_dict(data, SUPPORTED_VOICE_EFFECTS, n_slider_ticks, zero_idx=7)
    robot_effects_dict_list = get_effects_dict(data, SUPPORTED_ROBOT_EFFECTS, n_slider_ticks)

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
            voice_effects = get_current_effect_values(voice_effects_dict_list, i)
            robot_effects = get_current_effect_values(robot_effects_dict_list, i)

            spk_emb = torch.from_numpy(speaker_embedding.astype(np.float32))[None]

            tmp_wav_path = os.path.join(out_dir, bname + '_' + str(i) + '.wav')

            # TTS
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

            time_stretch = 1 - voice_effects['speed']
            pitch_factor = 1 - voice_effects['pitch_shift']
            pitch_range = 1 - voice_effects['pitch_range']

            assert pitch_range >= 0

            # Cast to Praat object
            sound = Sound(audio, sr)
            pitch = call(sound, "To Pitch", .01, 75, 600)
            manipulation = call([sound, pitch], "To Manipulation")
            pitch_values = pitch.selected_array["frequency"]

            idxs = np.where(pitch_values == 0)  # Remove NAs
            pitch_values = np.delete(pitch_values, idxs)
            time = np.delete(pitch.xs(), idxs)

            # Change pitch range
            pitch_values = pitch_values * pitch_factor

            # Scale pitc<h values
            full_range = pitch_values.max() - pitch_values.min()
            half_range = full_range / 2

            # Center all pitch values around 0
            pitch_rel = pitch_values - pitch_values.min() - half_range

            # Multiply with scalar and put it back to the original pitch height
            pitch_values = (pitch_rel * pitch_range) + pitch_values.min() + half_range

            pitch_tier = call(manipulation, "Extract pitch tier")
            # Make sure the pitch Tier is empty
            call(pitch_tier, "Remove points between", min(pitch.xs()) - 0.001, max(pitch.xs()) + 0.001)

            for i in range(len(pitch_values)):
                call(pitch_tier, "Add point", time[i], pitch_values[i])

            call([manipulation, pitch_tier], "Replace pitch tier")

            # Apply time stretch
            duration_tier = call("Create DurationTier", "tmp", 0, sound.xmax - sound.xmin)
            call([duration_tier], "Add point", 0, time_stretch)
            call([duration_tier, manipulation], "Replace duration tier")

            # Resynthesize
            sound = call(manipulation, "Get resynthesis (overlap-add)")
            call(sound, "Save as WAV file", tmp_wav_path)
            audio, sr = load(tmp_wav_path)

            additional_parameters = {
                'pitch_semitones': 5,
                'pitch_mirror': True,  # default mirror
                'griffin_iters': 0,  # highest compression

                # Vocoder settings
                'harmonics': 1.0,
                'esserintensity': 0.0,
                'chorus': 0.0,  # On or off, large effect
                'enveloperelease': 0.0,
                'vocoderband00': 0.0,
                'vocoderband01': 0.0,
                'vocoderband02': 0.0,
                'vocoderband03': 0.0,
                'vocoderband04': 0.0,
                'vocoderband05': 0.0,
                'vocoderband06': 0.0,
                'vocoderband07': 0.0,
                'vocoderband08': 0.0,
                'vocoderband09': 0.0,
                'vocoderband10': 0.0,
            }

            if robot_effects['flanger_type'] == 1:
                additional_parameters = {
                    **additional_parameters,
                    'flanger_delay': 1,
                    'flanger_depth': 10,
                    'flanger_frequency': 5,
                }
            elif robot_effects['flanger_type'] == 2:
                additional_parameters = {
                    **additional_parameters,
                    'flanger_delay': 0,
                    'flanger_depth': 50,
                    'flanger_frequency': 0,
                }
            elif robot_effects['flanger_type'] == 3:
                additional_parameters = {
                    **additional_parameters,
                    'flanger_delay': 20,
                    'flanger_depth': 20,
                    'flanger_frequency': 5,
                }
            elif robot_effects['flanger_type'] == 4:
                additional_parameters = {
                    **additional_parameters,
                    'flanger_delay': 1,
                    'flanger_depth': 10,
                    'flanger_frequency': 25,
                }
            elif robot_effects['flanger_type'] == 5:
                additional_parameters = {
                    **additional_parameters,
                    'flanger_delay': 10,
                    'flanger_depth': 0,
                    'flanger_frequency': 0,
                }

            if robot_effects['vocoder_type'] == 1:
                additional_parameters = {
                    **additional_parameters,
                    'vocoder_carrier_frequency': 10.0  #
                }
            elif robot_effects['vocoder_type'] == 2:
                additional_parameters = {
                    **additional_parameters,
                    'vocoder_carrier_frequency': 30.0  #
                }
            elif robot_effects['vocoder_type'] == 3:
                additional_parameters = {
                    **additional_parameters,
                    'vocoder_carrier_frequency': 60.0  #
                }

            elif robot_effects['vocoder_type'] == 4:
                additional_parameters = {
                    **additional_parameters,
                    'vocoder_carrier_frequency': 90.0  #
                }
            elif robot_effects['vocoder_type'] == 5:
                additional_parameters = {
                    **additional_parameters,
                    'vocoder_carrier_frequency': 120.0  #
                }

            fx = Fx(sr)
            del robot_effects['flanger_type']
            del robot_effects['vocoder_type']
            y = fx.process_audio(audio, robot_effects, additional_parameters)

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
                mimetype='audio/wav'
            ))
            response.headers.add('Access-Control-Allow-Origin', '*')
            response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
            response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')

            return response


# Render index.html
@app.route('/')
def index():
    return render_template('client.html')


if __name__ != "__main__":
    gunicorn_logger = logging.getLogger("gunicorn.error")
    app.logger.handlers = gunicorn_logger.handlers
    app.logger.setLevel(gunicorn_logger.level)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
