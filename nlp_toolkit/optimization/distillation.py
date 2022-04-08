from enum import Enum
from typing import List


class Criterion(object):
    def __init__(
        self,
        name: str = None,
        temperature: float = None,
        loss_types: List = None,
        loss_weight_ratio: List = None,
        layer_mappings: List = None,
        add_origin_loss: List = None
    ):
        self.name = name
        self.temperature = temperature
        self.loss_types = loss_types
        self.loss_weight_ratio = loss_weight_ratio
        self.layer_mappings = layer_mappings
        self.add_origin_loss = add_origin_loss


class DistillationCriterionMode(Enum):
    KNOWLEDGELOSS = "KnowledgeDistillationLoss"
    INTERMEDIATELAYERSLOSS = "IntermediateLayersKnowledgeDistillationLoss"


SUPPORTED_DISTILLATION_CRITERION_MODE = \
    set([approach.name for approach in DistillationCriterionMode])
