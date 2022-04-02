class Metric(object):
    def __init__(self, name: str, greater_is_better: bool = True, is_relative: bool = True,
                 criterion: float = 0.01, weight_ratio: float = None):
        self.name = name
        self.is_relative = is_relative
        self.criterion = criterion
        self.greater_is_better = greater_is_better
        self.weight_ratio = weight_ratio


class Objective(object):
    def __init__(self, name: str, greater_is_better: bool = True, weight_ratio: float = None):
        self.name = name
        self.greater_is_better = greater_is_better
        self.weight_ratio = weight_ratio

    @staticmethod
    def performance():
        return Objective(name="performance", greater_is_better=True)

    @staticmethod
    def modelsize():
        return Objective(name="modelsize", greater_is_better=False)


class DotDict(dict):
    def __init__(self):
        super(DotDict, self).__init__()
        self["performance"] = Objective(name="performance", greater_is_better=True)
        self["modelsize"] = Objective(name="modelsize", greater_is_better=False)

    def __getattr__(self, attr):
        return self.get(attr)

OBJECTIVES = DotDict()