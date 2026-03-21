# I/O
import pathlib
import tempfile
from librosa import load

# Processors
from parselmouth.praat import call
from parselmouth import Sound


SUPPORTED_VOICE_EFFECTS = ['speed']

def apply_voice_effects(audio, sr, voice_effects):
    # Cast to Praat object
    sound = Sound(audio, sr)
    pitch = call(sound, "To Pitch", .01, 75, 600)
    manipulation = call([sound, pitch], "To Manipulation")

    # Apply time stretch
    duration_tier = call("Create DurationTier", "tmp", 0, sound.xmax - sound.xmin)
    time_stretch = 1 - voice_effects['speed']
    call([duration_tier], "Add point", 0, time_stretch)
    call([duration_tier, manipulation], "Replace duration tier")

    # Resynthesize
    sound = call(manipulation, "Get resynthesis (overlap-add)")
    with tempfile.TemporaryDirectory() as out_dir:
        tmp_wav_path = pathlib.Path(out_dir)/"tmp.wav"
        call(sound, "Save as WAV file", tmp_wav_path)
        return load(tmp_wav_path)
