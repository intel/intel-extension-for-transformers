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


performance = Objective(name="performance", greater_is_better=True)
modelsize = Objective(name="modelsize", greater_is_better=False)
