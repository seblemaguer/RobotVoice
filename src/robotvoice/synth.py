import numpy as np

import warnings
warnings.simplefilter("ignore", UserWarning)

from robotvoice.dsp import normalize
from robotvoice.vits import VITS

synthesizer = VITS()


def synthesize(text, parameters) -> tuple[np.array, int]:

    # robot_effects_dict_list = get_effects_dict(parameters, SUPPORTED_ROBOT_EFFECTS, n_slider_ticks)
    # robot_effects = get_current_effect_values(robot_effects_dict_list, 0)
    # robot_effects, additional_parameters = update_parameters(robot_effects)

    # Run synthesizer
    audio, sr = synthesizer.synth(text)
    # # Apply effects
    # audio = Fx(sr).process_audio(audio, robot_effects, additional_parameters)

    # Normalize
    audio = normalize(audio)

    return audio, sr
