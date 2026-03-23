import numpy as np
import numpy.typing as npt


def normalize(audio: npt.NDArray, rms_level: int = -8) -> npt.NDArray:
    """
    Normalize the signal given a certain technique (peak or rms).
    Args:
        - audio    (np array) : waveform
        - rms_level (int) : rms level in dB.
    """
    # linear rms level and scaling factor
    r = 10 ** (rms_level / 10.0)
    a = np.sqrt((len(audio) * r**2) / np.sum(audio**2))

    # normalize
    return audio * a


def normalize_value(value: int | float, zero_idx: int, n_steps: int) -> float:
    assert value >= 0, f"The value should be positive but is {value}"
    n = n_steps - 1
    return (value / n) - (zero_idx / n)
