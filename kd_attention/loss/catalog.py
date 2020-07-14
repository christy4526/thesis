from .kdloss import KDExplanationLoss, KnowledgeDistillationLoss
from .multiloss_compatibles import CrossEntropyLoss


class Catalog:
    losses = {
        'KDExplanationLoss': KDExplanationLoss,
        'KnowledgeDistillationLoss': KnowledgeDistillationLoss,
        'CrossEntropyLoss': CrossEntropyLoss,
    }
