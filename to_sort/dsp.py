import os

import librosa.effects
import numpy as np
import math
from scipy import signal
from scipy import io
from pedalboard import Pedalboard, load_plugin

class Fx():
    def __init__(self, sr):
        self.framerate = sr
        self.fx_functions = {"flanger": self.flanger,
                             "distortion": self.ge_distortion,
                             "wahwah": self.ge_wahwah,
                             "tremolo": self.ge_tremolo,
                             "chorus": self.chorus,
                             "pitch": self.pitch,
                             "griffin": self.griffin,
                             "timestretch": self.timestretch,
                             "timeshift": self.timeshift,
                             "vocoder": self.vocoder}
        # vst3_path = os.path.join(os.path.expanduser("~"), '.VST3', 'TAL-Vocoder-2.vst3')
        self.vst_vocoder = load_plugin("./VSTs/TAL-Vocoder-2.vst3")
        self.vst_vocoder.inputmode = 1

    def generate_wave_input(self, freq, length, rate=44100, phase=0.0):
        length = int(length * rate)
        t = np.arange(length) / float(rate)
        omega = float(freq) * 2 * math.pi
        phase *= 2 * math.pi
        return omega * t + phase

    def sine(self, freq, length, rate=44100, phase=0.0):
        data = self.generate_wave_input(freq, length, rate, phase)
        return np.sin(data)

    def feedback_modulated_delay(self, data, modwave, dry, wet):
        out = data.copy()
        for i in range(len(data)):
            index = int(i - modwave[i])
            if index >= 0 and index < len(data):
                out[i] = out[i] * dry + out[index] * wet
        return out

    def flanger(self, data, frequency=1, dry=0.3, wet=0.7, depth=10.0, delay=1.0, additional_parameters=None):
        if(additional_parameters != None):
            if "flanger_frequency" in additional_parameters.keys():
                frequency = additional_parameters["flanger_frequency"]
            if "flanger_depth" in additional_parameters.keys():
                depth = additional_parameters["flanger_depth"]
            if "flanger_delay" in additional_parameters.keys():
                delay = additional_parameters["flanger_delay"]
        length = float(len(data)) / self.framerate
        mil = float(self.framerate) / 1000
        delay *= mil
        depth *= mil
        modwave = (self.sine(frequency, length) / 2 + 0.5) * depth + delay
        return self.feedback_modulated_delay(data, modwave, dry, wet)

    def modulated_delay(self, data, modwave, dry, wet):
        out = data.copy()
        for i in range(len(data)):
            index = int(i - modwave[i])
            if index >= 0 and index < len(data):
                out[i] = data[i] * dry + data[index] * wet
        return out

    def chorus(self, data, frequency=2, dry=0.5, wet=0.5, depth=0.9, delay=25.0, additional_parameters=None):
        if(additional_parameters != None):
            if "chorus_frequency" in additional_parameters.keys():
                frequency = additional_parameters["chorus_frequency"]
            if "chorus_depth" in additional_parameters.keys():
                depth = additional_parameters["chorus_depth"]
            if "chorus_delay" in additional_parameters.keys():
                delay = additional_parameters["chorus_delay"]
        length = float(len(data)) / self.framerate
        mil = float(self.framerate) / 1000
        delay *= mil
        depth *= mil
        modwave = (self.sine(frequency, length) / 2 + 0.5) * depth + delay
        return self.modulated_delay(data, modwave, dry, wet)

    def translate(self, input_signal, original_signal):
        # Figure out how 'wide' each range is
        leftMin = np.min(input_signal)
        leftMax = np.max(input_signal)
        rightMin = np.min(original_signal)
        rightMax = np.max(original_signal)

        leftSpan = leftMax - leftMin
        rightSpan = rightMax - rightMin

        # Convert the left range into a 0-1 range (float)
        valueScaled = np.float32(input_signal - leftMin) / np.float32(leftSpan)
        ret = (valueScaled * rightSpan) + rightMin
        # Convert the 0-1 range into a value in the right range.
        return ret

    def norm_signal(self, input_signal):
        output_signal = input_signal / np.max(np.absolute(input_signal))
        return output_signal

    def ge_tremolo(self, input_signal, alpha=1, modfreq=10, additional_parameters=None):
        if (additional_parameters != None):
            if "tremolo_alpha" in additional_parameters.keys():
                alpha = additional_parameters["tremolo_alpha"]
            if "tremolo_modfreq" in additional_parameters.keys():
                modfreq = additional_parameters["tremolo_modfreq"]
        output_signal = np.zeros(len(input_signal))
        for n in range(len(input_signal)):
            trem = 1 + alpha * np.sin(2 * np.pi * modfreq * n / self.framerate)
            output_signal[n] = trem * input_signal[n]
        return output_signal

    def ge_distortion(self, input_signal, alpha=5, additional_parameters=None):
        if (additional_parameters != None):
            if "distortion_alpha" in additional_parameters.keys():
                alpha = additional_parameters["distortion_alpha"]
        q = np.sign(input_signal)
        output_signal = q * (1 - np.exp(alpha * q * input_signal))
        output_signal = self.translate(output_signal, input_signal)
        return output_signal

    def ge_wahwah(self, input_signal, damp=0.49, minf=100.0, maxf=2000.0, wahf=2000.0, additional_parameters=None):
        if (additional_parameters != None):
            if "wahwah_damp" in additional_parameters.keys():
                damp = additional_parameters["wahwah_damp"]
            if "wahwah_minf" in additional_parameters.keys():
                minf = additional_parameters["wahwah_minf"]
            if "wahwah_maxf" in additional_parameters.keys():
                maxf = additional_parameters["wahwah_maxf"]
            if "wahwah_wahf" in additional_parameters.keys():
                wahf = additional_parameters["wahwah_wahf"]
        output_signal = np.zeros(len(input_signal))
        outh = np.zeros(len(input_signal))
        outl = np.zeros(len(input_signal))
        delta = wahf / self.framerate
        centerf = np.concatenate((np.arange(minf, maxf, delta), np.arange(maxf, minf, -delta)))
        while len(centerf) < len(input_signal):
            centerf = np.concatenate((centerf, centerf))
        centerf = centerf[:len(input_signal)]
        f1 = 2 * np.sin(np.pi * centerf[0] / self.framerate)
        outh[0] = input_signal[0]
        output_signal[0] = f1 * outh[0]
        outl[0] = f1 * output_signal[0]
        for n in range(1, len(input_signal)):
            outh[n] = input_signal[n] - outl[n - 1] - 2 * damp * output_signal[n - 1]
            output_signal[n] = f1 * outh[n] + output_signal[n - 1]
            outl[n] = f1 * output_signal[n] + outl[n - 1]
            f1 = 2 * np.sin(np.pi * centerf[n] / self.framerate)
        return output_signal

    def pitch(self, input_signal, semitones=12.0, mirror=False, additional_parameters=None):
        if (additional_parameters != None):
            if "pitch_semitones" in additional_parameters.keys():
                semitones = additional_parameters["pitch_semitones"]
            if "pitch_mirror" in additional_parameters.keys():
                mirror = additional_parameters["pitch_mirror"]
        y = librosa.effects.pitch_shift(input_signal, sr=self.framerate,n_steps=semitones, bins_per_octave=12,res_type='kaiser_best')

        if mirror:
            z = librosa.effects.pitch_shift(input_signal, sr=self.framerate, n_steps=-semitones, bins_per_octave=12,
                                            res_type='kaiser_best')
            y = (y + z)/2.0
        return y

    def griffin(self, input_signal, iters=0, additional_parameters=None):
        if (additional_parameters != None):
            if "griffin_iters" in additional_parameters.keys():
                iters = additional_parameters["griffin_iters"]
        D = librosa.stft(input_signal)
        spect, _ = librosa.magphase(D)
        audio_signal = librosa.griffinlim(spect, n_iter=iters)
        pad = np.zeros((len(input_signal)-len(audio_signal)))
        audio_signal = np.append(audio_signal,pad)
        return audio_signal

    def timestretch(self, input_signal, speed= 1.0):
        audio_signal = librosa.effects.time_stretch(input_signal, rate=speed)
        return audio_signal

    def timeshift(self, input_signal, shift_ms=10):
        padding_signal_length = (self.framerate / 1000) * shift_ms
        padding_signal = np.zeros(int(padding_signal_length),dtype=np.float32)
        s1 = np.concatenate((padding_signal, input_signal), axis=0)
        s2 = np.concatenate(( input_signal, padding_signal), axis=0)
        out = np.add(s1,s2) / 2.0
        return out

    def vocoder(self, input_signal, additional_parameters=None):
        # if "volume" in additional_parameters.keys():
        #     self.vst_vocoder.volume = additional_parameters["volume"]
        # if "noisevolume" in additional_parameters.keys():
        #     self.noisevolume.noisevolume = additional_parameters["noisevolume"]
        # if "pulsevolume" in additional_parameters.keys():
        #     self.vst_vocoder.pulsevolume = additional_parameters["pulsevolume"]
        # if "sawvolume" in additional_parameters.keys():
        #     self.vst_vocoder.sawvolume = additional_parameters["sawvolume"]
        # if "suboscvolume" in additional_parameters.keys():
        #     self.vst_vocoder.suboscvolume = additional_parameters["suboscvolume"]
        # if "osctranspose" in additional_parameters.keys():
        #     self.vst_vocoder.osctranspose = additional_parameters["osctranspose"]
        # if "suboscoctave" in additional_parameters.keys():
        #     self.vst_vocoder.suboscoctave = additional_parameters["suboscoctave"]
        # if "oscsync" in additional_parameters.keys():
        #     self.vst_vocoder.oscsync = additional_parameters["oscsync"]
        # if "pulsetune" in additional_parameters.keys():
        #     self.vst_vocoder.pulsetune = additional_parameters["pulsetune"]
        # if "sawtune" in additional_parameters.keys():
        #     self.vst_vocoder.sawtune = additional_parameters["sawtune"]
        # if "pulsefinetune" in additional_parameters.keys():
        #     self.vst_vocoder.pulsefinetune = additional_parameters["pulsefinetune"]
        # if "sawfinetune" in additional_parameters.keys():
        #     self.noisevolume.sawfinetune = additional_parameters["sawfinetune"]
        # if "polymode" in additional_parameters.keys():
        #     self.vst_vocoder.polymode = additional_parameters["polymode"]
        # if "portamento" in additional_parameters.keys():
        #     self.vst_vocoder.portamento = additional_parameters["portamento"]
        # if "tune" in additional_parameters.keys():
        #     self.vst_vocoder.tune = additional_parameters["tune"]
        # if "panic" in additional_parameters.keys():
        #     self.vst_vocoder.panic = additional_parameters["panic"]
        if "harmonics" in additional_parameters.keys():
            self.vst_vocoder.harmonics = additional_parameters["harmonics"]
        if "esserintensity" in additional_parameters.keys():
            self.vst_vocoder.esserintensity = additional_parameters["esserintensity"]
        if "chorus" in additional_parameters.keys():
            self.vst_vocoder.chorus = additional_parameters["chorus"]
        if "enveloperelease" in additional_parameters.keys():
            self.vst_vocoder.enveloperelease = additional_parameters["enveloperelease"]
        if "vocoderband00" in additional_parameters.keys():
            self.vst_vocoder.vocoderband00 = additional_parameters["vocoderband00"]
        if "vocoderband01" in additional_parameters.keys():
            self.vst_vocoder.vocoderband01 = additional_parameters["vocoderband01"]
        if "vocoderband02" in additional_parameters.keys():
            self.vst_vocoder.vocoderband02 = additional_parameters["vocoderband02"]
        if "vocoderband03" in additional_parameters.keys():
            self.vst_vocoder.vocoderband03 = additional_parameters["vocoderband03"]
        if "vocoderband04" in additional_parameters.keys():
            self.vst_vocoder.vocoderband04 = additional_parameters["vocoderband04"]
        if "vocoderband05" in additional_parameters.keys():
            self.vst_vocoder.vocoderband05 = additional_parameters["vocoderband05"]
        if "vocoderband06" in additional_parameters.keys():
            self.vst_vocoder.vocoderband06 = additional_parameters["vocoderband06"]
        if "vocoderband07" in additional_parameters.keys():
            self.vst_vocoder.vocoderband07 = additional_parameters["vocoderband07"]
        if "vocoderband08" in additional_parameters.keys():
            self.vst_vocoder.vocoderband08 = additional_parameters["vocoderband08"]
        if "vocoderband09" in additional_parameters.keys():
            self.vst_vocoder.vocoderband09 = additional_parameters["vocoderband09"]
        if "vocoderband10" in additional_parameters.keys():
            self.vst_vocoder.vocoderband10 = additional_parameters["vocoderband10"]
        if "vocoder_carrier_frequency" in additional_parameters.keys():
            cf = additional_parameters["vocoder_carrier_frequency"]
        else:
            cf = 60

        sr = int(self.framerate)
        t = np.linspace(0, 1, sr, endpoint=False)
        t = signal.square(2 * np.pi * int(cf) * t) * 0.2

        len_input_signal = len(input_signal)
        input_signal_times = math.floor(len_input_signal / sr)

        rest = []
        m = 0
        for n in range(input_signal_times * sr, len_input_signal):
            rest.append(t[m])
            m+=1
        rest = np.asarray(rest)

        tmp_signal = t
        if input_signal_times > 1:
            for i in range(0, input_signal_times - 1):
                tmp_signal = np.concatenate((tmp_signal, t))
        if len(rest) > 0:
            tmp_signal = np.concatenate((tmp_signal, rest))
        sig_len = min(len(tmp_signal), len(input_signal))
        stereo = np.stack((tmp_signal[:sig_len], input_signal[:sig_len]), 0)

        effected = self.vst_vocoder(stereo, sr)
        y = np.float32(effected)
        out = np.add(y[0], y[1]) / 2.0
        return out

    def process_audio(self, input_data, fx_chain, additional_parameters=None):
        y = input_data
        for fx, l in fx_chain.items():
            if l > 0.0:
                if fx == "timestretch":
                    y = np.float32(self.fx_functions[fx](y, speed= l))
                elif fx == "timeshift":
                    y = np.float32(self.fx_functions[fx](y, shift_ms=l))
                else:
                    y = np.float32(((1.0 - l) * y + l * self.fx_functions[fx](y, additional_parameters=additional_parameters)))
        return y
