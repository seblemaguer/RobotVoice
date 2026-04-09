
import numpy.typing as npt

class Synthesizer:
    def __init__(self):
        pass

    def synth(self, text: str) -> tuple[npt.NDArray, int]:
        raise NotImplementedError("This is an abstract method which should be overriden!")
