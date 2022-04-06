class LazyImport(object):
    """Lazy import python module till use

       Args:
           module_name (string): The name of module imported later
    """

    def __init__(self, module_name):
        self.module_name = module_name
        self.module = None

    def __getattr__(self, name):
        if self.module is None:
            # __import__ returns top level module
            top_level_module = __import__(self.module_name)
            if len(self.module_name.split('.')) == 1:
                self.module = top_level_module
            else:
                # for cases that input name is foo.bar.module
                module_list = self.module_name.split('.')
                temp_module = top_level_module
                for temp_name in module_list[1:]:
                    temp_module = getattr(temp_module, temp_name)
                self.module = temp_module

        return getattr(self.module, name)