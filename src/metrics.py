from typing import List

import torch
from torchmetrics import Metric


class PrecisionAtOne(Metric):
    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.add_state("correct", default=torch.tensor(0, dtype=torch.float), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0, dtype=torch.float), dist_reduce_fx="sum")

    def update(self, preds: List[str], target: List[List[str]]):

        for id, top_prediction in enumerate(preds):
            if top_prediction in target[id]:
                self.correct += 1
        self.total += len(target)

    def compute(self):
        return self.correct.float() / self.total


def precision_at_k(predicted: List[str], gold: List[str], k: int) -> float:
    num = len([x for x in predicted[:k] if x in gold])
    den = len(predicted[:k])
    if den == 0:
        return 0
    return num / den


def recall_at_k(predicted: List[str], gold: List[str], k: int) -> float:
    num = len([x for x in predicted[:k] if x in gold])
    den = len(gold[:k])
    return num / den