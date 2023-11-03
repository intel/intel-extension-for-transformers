import torch
from torch.nn import CrossEntropyLoss, MSELoss


class ConstLambdaLoss():

    def __init__(self, loss_lambda: float = 0.0) -> None:
        self.loss_lambda = float(loss_lambda)
        self.cross_entropy = CrossEntropyLoss(reduce=False)

    def __call__(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        loss_list = []
        has_been_learned = torch.zeros((logits.shape[0], target.shape[0]), device=logits.device)
        for i in range(logits.shape[0]):
            current_logits = logits[i, ...]
            loss = self.cross_entropy(current_logits, target)
            if i > 0:
                loss = loss * (1 - self.loss_lambda * torch.mean(has_been_learned[0:i, ...], axis=0))
            loss = torch.mean(loss)
            loss_list.append(loss)
            predictions = torch.argmax(current_logits, axis=1)
            has_been_learned[i] = (predictions == target).detach()
        return torch.sum(torch.stack(loss_list, dim=0))


class BaselineLoss():
    def __init__(self):
        self.cross_entropy = CrossEntropyLoss(reduce=True)
        self.lte_loss_fct = MSELoss()

    def __call__(self, logits: torch.Tensor, target: torch.Tensor, lte_logits: torch.Tensor = None) -> torch.Tensor:
        loss_list = []
        lte_loss_list = []

        for i in range(logits.shape[0]):
            loss = torch.mean(self.cross_entropy(logits[i, ...], target))
            loss_list.append(loss)

            # LTE loss
            lte_gold = torch.eq(
                torch.argmax(logits[i, ...], dim=1),
                target
            )  # 0 for wrong/continue, 1 for right/exit
            exit_label = lte_gold.float().unsqueeze(1)
            loss = torch.mean(self.lte_loss_fct(lte_logits[i, ...], exit_label))
            lte_loss_list.append(loss)

        return torch.sum(torch.stack(lte_loss_list, dim=0)) + torch.sum(torch.stack(loss_list, dim=0))


class ConstLambdaLossOnes():

    def __init__(self, loss_lambda: float = 0.0) -> None:
        self.loss_lambda = float(loss_lambda)
        self.cross_entropy = CrossEntropyLoss(reduce=False)

    def __call__(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        loss_list = []
        has_been_learned = torch.ones((logits.shape[0], target.shape[0]), device=logits.device)
        for i in range(logits.shape[0]):
            current_logits = logits[i, ...]
            loss = self.cross_entropy(current_logits, target)
            if i > 0:
                loss = loss * (1 - self.loss_lambda * torch.mean(has_been_learned[0:i, ...], axis=0))
            loss = torch.mean(loss)
            loss_list.append(loss)
        return torch.sum(torch.stack(loss_list, dim=0))


class DynamicLambdaLoss():

    def __init__(self, lambda_func: callable = lambda x: x) -> None:
        self.lambda_func = lambda_func
        self.cross_entropy = CrossEntropyLoss(reduce=False)

    def __call__(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        loss_list = []
        has_been_learned = torch.zeros((logits.shape[0], target.shape[0]), device=logits.device)
        for i in range(logits.shape[0]):
            current_logits = logits[i, ...]
            loss = self.cross_entropy(current_logits, target)
            if i > 0:
                loss = loss * (1 - self.loss_lambda * torch.mean(has_been_learned[0:i, ...], axis=0))
            loss = torch.mean(loss)
            loss_list.append(loss)
            predictions = torch.argmax(current_logits, axis=1)
            has_been_learned[i] = (predictions == target).detach()
        return torch.sum(torch.stack(loss_list, dim=0))


class ConstLambdaLossWithOr():

    def __init__(self, loss_lambda: float = 0.0) -> None:
        self.loss_lambda = float(loss_lambda)
        self.cross_entropy = CrossEntropyLoss(reduce=False)

    def __call__(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        loss_list = []
        has_been_learned = torch.zeros((logits.shape[0], target.shape[0]), device=logits.device)
        for i in range(logits.shape[0]):
            current_logits = logits[i, ...]
            loss = self.cross_entropy(current_logits, target)
            if i > 0:
                loss = loss * (1 - self.loss_lambda * torch.max(has_been_learned[0:i, ...], axis=0).values)
            loss = torch.mean(loss)
            loss_list.append(loss)
            predictions = torch.argmax(current_logits, axis=1)
            has_been_learned[i] = (predictions == target).detach()
        return torch.sum(torch.stack(loss_list, dim=0))


class ConstLambdaLossPrevLayer():

    def __init__(self, loss_lambda: float = 0.0) -> None:
        self.const_lambda_loss = ConstLambdaLoss(loss_lambda)

    def __call__(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.const_lambda_loss(logits[-2, ...])


LOSS_MAP = {
    'baseline': BaselineLoss,
    'const_lambda': ConstLambdaLoss,
    'const_lambda_or': ConstLambdaLossWithOr,
    'const_lambda_ones': ConstLambdaLossOnes,
    'const_lambda_prev_layer': ConstLambdaLossPrevLayer
}


class MultiExitCrossEntropyLoss():
    def __init__(self):
        self.cross_entropy = CrossEntropyLoss(reduce=True)

    def __call__(self, logits: torch.Tensor, target: torch.Tensor, lte_logits: torch.Tensor = None,
                 gold_classifier=None) -> torch.Tensor:
        loss_list = []

        for i in range(logits.shape[0]):
            loss = torch.mean(self.cross_entropy(logits[i, ...], target))
            loss_list.append(loss)

        if gold_classifier is not None:
            return torch.Tensor(loss_list[gold_classifier])
        return torch.sum(torch.stack(loss_list, dim=0))


class MultiExitMSELossForLTE():
    def __init__(self):
        self.lte_loss_fct = MSELoss()

    def __call__(self, logits: torch.Tensor, target: torch.Tensor, lte_logits: torch.Tensor = None,
                 gold_classifier=None) -> torch.Tensor:
        lte_loss_list = []

        for i in range(logits.shape[0]):
            lte_gold = torch.eq(torch.argmax(logits[i, ...], dim=1), target).float().unsqueeze(
                1)  # 0 for wrong/continue, 1 for right/exit
            loss = torch.mean(self.lte_loss_fct(lte_logits[i, ...], lte_gold))
            lte_loss_list.append(loss)

        if gold_classifier is not None:
            return torch.Tensor(lte_loss_list[gold_classifier])
        return torch.sum(torch.stack(lte_loss_list, dim=0))
