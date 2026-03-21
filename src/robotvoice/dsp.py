import numpy as np


def normalize(audio: np.array, rms_level=-6) -> np.array:
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


def normalize_value(effect_name: str, value: int | float, zero_idx: int, n_slider_ticks: int) -> float:
    assert value >= 0
    assert value < n_slider_ticks, f"Effect {effect_name} must be < {n_slider_ticks}"
    n = n_slider_ticks - 1
    return (value / n) - (zero_idx / n)
