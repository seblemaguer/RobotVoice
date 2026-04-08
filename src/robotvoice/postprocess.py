import logging
from pathlib import Path
from pedalboard import load_plugin, Pedalboard
import numpy.typing as npt
import numpy as np


class PostProcessor:
    def __init__(self, vst_plugin_paths: Path | list[Path]):
        self._logger = logging.getLogger(self.__class__.__name__)
        # Instantiate plugins
        self._plugins = dict()
        if isinstance(vst_plugin_paths, list):
            for plugin_path in vst_plugin_paths:
                if plugin_path.suffix != ".vst3":
                    raise Exception(f"the path {plugin_path} is not a valid VST3 plugin")

                self._plugins[plugin_path.stem] = load_plugin(str(plugin_path.resolve()))
        else:
            if vst_plugin_paths.suffix == ".vst3":
                self._plugins[vst_plugin_paths.stem] = load_plugin(str(vst_plugin_paths.resolve()))
            else:

                for plugin_path in vst_plugin_paths.glob("*.vst3"):
                    self._plugins[plugin_path.stem] = load_plugin(str(plugin_path.resolve()))

                if len(self._plugins) == 0:
                    raise Exception(f"Didn't find VST3 plugins in {vst_plugin_paths}")

    def list_available_plugins(self) -> list[str]:
        return list(self._plugins.keys())

    def get_parameter_value(self, plugin_name: str, parameter_name: str) -> float | bool:

        # Ensure the plugin and the parameter are valid
        assert (
            plugin_name in self._plugins
        ), f"{plugin_name} is not part of the known plugins: {self.list_available_plugins()}"
        assert (
            parameter_name in self._plugins[plugin_name].parameters
        ), f"{parameter_name} is not a parameter from: {list(self._plugins[plugin_name].parameters.keys())}"

        return self._plugins[plugin_name].parameters[parameter_name].raw_value

    def configure_plugin(self, plugin_name: str, plugin_conf: dict[str, float | bool | str]):
        assert (
            plugin_name in self._plugins
        ), f"{plugin_name} is not part of the known plugins: {self.list_available_plugins()}"

        for p_name, p_value in plugin_conf.items():
            param = self._plugins[plugin_name].parameters[p_name]
            if param.type == float:
                param.raw_value = float(p_value)
            elif param.type == bool:
                param.raw_value = bool(p_value)
            else:
                self._logger.warning(f'Ignoring parameter [{p_name}] as type "{param.type}" is not supported yet')

    def list_info_plugin(
        self, plugin_name: str
    ) -> dict[str, tuple[float, float, float, float, float] | tuple[bool, bool]]:
        parameters = dict()

        # Get plugin
        cur_plugin = self._plugins[plugin_name]
        for p_name, p_info in cur_plugin.parameters.items():
            if p_info.type == bool:
                parameters[p_name] = (p_info.default_raw_value, p_info.raw_value)
            elif p_info.type == float:
                if (p_info.min_value == -np.inf) or (p_info.max_value == np.inf):
                    self._logger.warning(
                        f"The type of the parameter {p_name} relies on infinite values, that is not supported, so we ignore this parameter"
                    )
                else:
                    parameters[p_name] = (
                        p_info.min_value,
                        p_info.approximate_step_size if p_info.step_size is None else p_info.step_size,
                        p_info.max_value,
                        p_info.default_raw_value,
                        p_info.raw_value,
                    )
            elif p_info.type == str:
                self._logger.warning(f"The type of the parameter {p_name} is str which is ignored for now")
            else:
                raise Exception(f"The type of the parameter {p_name} is unknown: {p_info.type}")

        return parameters

    def get_effects(self) -> dict[str, dict[str, tuple[float, float, float, float, float] | tuple[bool, bool]]]:
        dict_conf = {}
        list_plugins = self.list_available_plugins()
        for plugin_name in list_plugins:
            dict_conf[plugin_name] = self.list_info_plugin(plugin_name)

        return dict_conf

    def apply(self, plugins: list[str], audio: npt.NDArray, sr: int) -> npt.NDArray:

        # Make a pretty interesting sounding guitar pedalboard:
        board = Pedalboard([self._plugins[cur_plugin] for cur_plugin in plugins])

        # Run the audio through this pedalboard!
        effected = board(audio, sr)

        return effected
