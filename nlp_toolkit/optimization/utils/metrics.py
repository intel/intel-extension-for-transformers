class Metric(object):
    def __init__(self, name: str, greater_is_better: bool = True, is_relative: bool = True,
                 criterion: float = 0.01, weight_ratio: float = None):
        self.name = name
        self.is_relative = is_relative
        self.criterion = criterion
        self.greater_is_better = greater_is_better
        self.weight_ratio = weight_ratio
