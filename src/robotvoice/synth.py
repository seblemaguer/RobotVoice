import numpy.typing as npt

import warnings

warnings.simplefilter("ignore", UserWarning)

from robotvoice.dsp import normalize
from robotvoice.effects.robot import (
    get_effects_dict,
    SUPPORTED_ROBOT_EFFECTS,
    get_current_effect_values,
    update_parameters,
)
from robotvoice.effects.fx import Fx
from robotvoice.vits import VITS

synthesizer = VITS()


def synthesize(text: str, parameters: dict | None) -> tuple[npt.NDArray, int]:

    # Run synthesizer
    audio, sr = synthesizer.synth(text)

    # Apply robot effects
    if (parameters is not None) and parameters and ("robot_effect" in parameters) and (parameters["robot_effect"]):
        robot_effects_dict_list = get_effects_dict(parameters, SUPPORTED_ROBOT_EFFECTS)
        robot_effects = get_current_effect_values(robot_effects_dict_list, 0)
        robot_effects, additional_parameters = update_parameters(robot_effects)
        audio = Fx(sr).process_audio(audio, robot_effects, additional_parameters)

    # Normalize
    audio = normalize(audio)

    return audio, sr
