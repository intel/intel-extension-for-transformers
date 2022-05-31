import importlib


class LazyImport(object):
    """Lazy import python module till use

       Args:
           module_name (string): The name of module imported later
    """

    def __init__(self, module_name):
        self.module_name = module_name
        self.module = None

    def __getattr__(self, name):
        try:
            self.module = importlib.import_module(self.module_name)
            mod = getattr(self.module, name)
        except:
            spec = importlib.util.find_spec(str(self.module_name + '.' + name))
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
        return mod

    def __call__(self, *args, **kwargs):
        self.module = importlib.import_module(self.module_name)
        return self.module(*args, **kwargs)
