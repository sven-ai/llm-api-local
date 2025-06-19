import importlib
import os
import sys

import yaml

_cloud_src = "/sven/src"
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), _cloud_src))
)


class _Module:
    def __init__(self, file_name):
        self.file_name = file_name
        self.config = self._load_config()
        # self.module = self._load_module()

    def _load_config(self):
        def load_yaml(path):
            with open(path, "r") as file:
                print(f"Loading yaml config: {path}")
                return yaml.safe_load(file)

        parent = f"{_cloud_src}/{self.file_name}"
        path = parent if os.path.exists(parent) else self.file_name
        return load_yaml(path)

    def _load_module(self):
        """Dynamically load the specified module."""
        module_name = self.config["active"]
        try:
            # Import the module dynamically
            module = importlib.import_module(module_name)
            # Get the class name (assuming it's the capitalized module name)
            class_name = module_name.split("_")[0].capitalize()
            # print(f'class_name: {class_name}')
            # print(f'module: {module}')
            # Get the class from the module
            module_class = getattr(module, class_name)
            return module_class()
        except (ImportError, AttributeError) as e:
            raise ImportError(f"Failed to load module {module_name}: {str(e)}")


def load_module(yml_path: str):
    try:
        return _Module(yml_path)._load_module()
    except Exception as e:
        print(f"Failed to load module from yml: {yml_path}. Exc: {str(e)}")


def load_config(yml_path: str):
    try:
        return _Module(yml_path).config
    except Exception as e:
        print(f"Failed to load config from yml: {yml_path}. Exc: {str(e)}")
