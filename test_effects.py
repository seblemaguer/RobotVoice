from dsp import RobotFx

from librosa import load
import soundfile as sf


infile = 'test_vits.wav'
outfile = 'my_processed_audio_file.wav'

x, sr = load(infile)

fx_factors = {"flanger": 0.5,
           "distortion": 0,
           "wahwah": 0.0,
           "tremolo": 0.5,
           "chorus": 0.0,
           "octaveup": 0.5}

fx = RobotFx(sr)
y = fx.process_audio(x, fx_factors)

sf.write(outfile, y, sr)
