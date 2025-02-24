
import yaml
import importlib

class LoadModule:
    def __init__(self, config_path):
        self.config_path = config_path
        self.config = self._load_config()
        self.module = self._load_module()
    
    def _load_config(self):
        with open(self.config_path, 'r') as file:
            return yaml.safe_load(file)
    
    def _load_module(self):
        """Dynamically load the specified module."""
        module_name = self.config['active']
        try:
            # Import the module dynamically
            module = importlib.import_module(module_name)
            # Get the class name (assuming it's the capitalized module name)
            class_name = module_name.split('_')[0].capitalize()
            # print(f'class_name: {class_name}')
            # print(f'module: {module}')
            # Get the class from the module
            module_class = getattr(module, class_name)
            return module_class()
        except (ImportError, AttributeError) as e:
            raise ImportError(f"Failed to load module {module_name}: {str(e)}")
    



def load_module(yml_path: str):
	try:
		return LoadModule(yml_path).module
	except Exception as e:
	    print(f"Failed to load module from yml: {yml_path}. Exc: {str(e)}")


