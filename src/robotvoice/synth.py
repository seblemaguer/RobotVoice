import numpy.typing as npt

import warnings

warnings.simplefilter("ignore", UserWarning)

from robotvoice.synthesizer.vits import DistantVITS

synthesizer = DistantVITS()


def synthesize(text: str, parameters: dict | None) -> tuple[npt.NDArray, int]:

    # Run synthesizer
    audio, sr = synthesizer.synth(text)

    return audio, sr
