import numpy.typing as npt

import warnings

warnings.simplefilter("ignore", UserWarning)

from robotvoice.vits import VITS

synthesizer = VITS()


def synthesize(text: str, parameters: dict | None) -> tuple[npt.NDArray, int]:

    # Run synthesizer
    audio, sr = synthesizer.synth(text)

    return audio, sr
