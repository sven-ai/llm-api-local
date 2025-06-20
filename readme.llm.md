This is a python-based project.

It has a pluggable structure. Loaded code is called modules, loading handled by `loader.py`. Usually there is a `<modulename>.yml` config file that defines a module to load via `active` param value - the value is a name of the `<modulename>.py` file to be loaded as module.
