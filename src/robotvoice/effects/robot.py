from robotvoice.dsp import normalize_value

ROBOT_EFFECT_TICKS = {
    "pitch": [0, 0.08, 0.16, 0.25, 0.27, 0.29, 0.31, 0.33, 0.35, 0.37, 0.40, 0.42, 0.44, 0.46, 0.48, 0.50],
    "tremolo": [0, 0.07, 0.13, 0.20, 0.22, 0.24, 0.25, 0.27, 0.29, 0.31, 0.32, 0.34, 0.36, 0.37, 0.39, 0.40],
    "flanger": [0, 0.10, 0.21, 0.32, 0.38, 0.43, 0.47, 0.52, 0.56, 0.59, 0.62, 0.65, 0.69, 0.72, 0.74, 0.78],
    "griffin": [0, 0.12, 0.24, 0.35, 0.40, 0.45, 0.50, 0.55, 0.61, 0.68, 0.75, 0.81, 0.86, 0.91, 0.95, 1.00],
    "timeshift": [0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 36, 39, 42, 45],
    "flanger_type": [1, 2, 3, 4, 5],
    "vocoder": [0, 0.03, 0.06, 0.08, 0.10, 0.12, 0.14, 0.15, 0.18, 0.21, 0.24, 0.25, 0.28, 0.31, 0.34, 0.35],
    "vocoder_type": [1, 2, 3, 4, 5],
}
N_STEPS = len(ROBOT_EFFECT_TICKS["pitch"])
SUPPORTED_ROBOT_EFFECTS = list(ROBOT_EFFECT_TICKS.keys())


def get_slider_value(effect_name, value, zero_idx):
    if effect_name in ROBOT_EFFECT_TICKS:
        return ROBOT_EFFECT_TICKS[effect_name][value]
    else:
        return normalize_value(value, zero_idx, N_STEPS)


def get_effects_dict(data, supported_effects, zero_idx=0):
    effects = {}
    for effect_name in supported_effects:
        assert effect_name in data, f"Effect {effect_name} not found in data"
        value = data[effect_name]

        if type(value) == str:
            value = eval(value)

        if type(value) == list:
            value = [get_slider_value(effect_name, v, zero_idx) for v in value]
        elif type(value) == int or type(value) == float:
            value = [get_slider_value(effect_name, value, zero_idx)]
        else:
            raise ValueError(f"Effect {effect_name} has invalid value {value}")

        effects[effect_name] = value
    return effects


def get_current_effect_values(effect_dict_list, index) -> dict:
    return {k: v[index] for k, v in effect_dict_list.items()}


def update_parameters(
    robot_effects: dict[str, int | bool | float],
) -> tuple[dict[str, int | bool | float], dict[str, int | bool | float]]:
    additional_parameters = {
        "pitch_semitones": 5,
        "pitch_mirror": True,  # default mirror
        "griffin_iters": 0,  # highest compression
        # Vocoder settings
        "harmonics": 1.0,
        "esserintensity": 0.0,
        "chorus": 0.0,  # On or off, large effect
        "enveloperelease": 0.0,
        "vocoderband00": 0.0,
        "vocoderband01": 0.0,
        "vocoderband02": 0.0,
        "vocoderband03": 0.0,
        "vocoderband04": 0.0,
        "vocoderband05": 0.0,
        "vocoderband06": 0.0,
        "vocoderband07": 0.0,
        "vocoderband08": 0.0,
        "vocoderband09": 0.0,
        "vocoderband10": 0.0,
    }

    if robot_effects["flanger_type"] == 1:
        additional_parameters = {
            **additional_parameters,
            "flanger_delay": 1,
            "flanger_depth": 10,
            "flanger_frequency": 5,
        }
    elif robot_effects["flanger_type"] == 2:
        additional_parameters = {
            **additional_parameters,
            "flanger_delay": 0,
            "flanger_depth": 50,
            "flanger_frequency": 0,
        }
    elif robot_effects["flanger_type"] == 3:
        additional_parameters = {
            **additional_parameters,
            "flanger_delay": 20,
            "flanger_depth": 20,
            "flanger_frequency": 5,
        }
    elif robot_effects["flanger_type"] == 4:
        additional_parameters = {
            **additional_parameters,
            "flanger_delay": 1,
            "flanger_depth": 10,
            "flanger_frequency": 25,
        }
    elif robot_effects["flanger_type"] == 5:
        additional_parameters = {
            **additional_parameters,
            "flanger_delay": 10,
            "flanger_depth": 0,
            "flanger_frequency": 0,
        }

    if robot_effects["vocoder_type"] == 1:
        vocoder_carrier_frequency = 10.0
    elif robot_effects["vocoder_type"] == 2:
        vocoder_carrier_frequency = 30.0
    elif robot_effects["vocoder_type"] == 3:
        vocoder_carrier_frequency = 60.0
    elif robot_effects["vocoder_type"] == 4:
        vocoder_carrier_frequency = 90.0
    elif robot_effects["vocoder_type"] == 5:
        vocoder_carrier_frequency = 120.0
    else:
        raise ValueError("Invalid vocoder type")

    additional_parameters = {**additional_parameters, "vocoder_carrier_frequency": vocoder_carrier_frequency}
    del robot_effects["flanger_type"]
    del robot_effects["vocoder_type"]
    return robot_effects, additional_parameters
