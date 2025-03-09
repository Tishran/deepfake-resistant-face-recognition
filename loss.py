from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from torch import Tensor
from torch.nn import Module, TripletMarginWithDistanceLoss
from torch.nn import functional as F

from oml.losses import TripletLossWithMiner
from oml.functional.losses import get_reduced
from oml.interfaces.criterions import ITripletLossWithMiner
from oml.interfaces.miners import ITripletsMiner, labels2list
from oml.miners.cross_batch import TripletMinerWithMemory
from oml.miners.inbatch_all_tri import AllTripletsMiner
from oml.utils.misc_torch import elementwise_dist

TLogs = Dict[str, float]


def cosine_distance(x1: torch.Tensor, x2: torch.Tensor):
    return 1 - F.cosine_similarity(x1, x2)


class CosineTripletLossWithMiner(TripletLossWithMiner):

    criterion_name = "cose triplet"  # for better logging

    def __init__(
        self,
        margin: Optional[float],
        miner: ITripletsMiner = AllTripletsMiner(),
        reduction: str = "mean",
        need_logs: bool = False,
    ):
        assert reduction in ("mean", "sum", "none")
        assert (margin is None) or (margin > 0)

        super(CosineTripletLossWithMiner, self).__init__(
            margin, miner, reduction, need_logs
        )
        self.tri_loss = TripletMarginWithDistanceLoss(
            distance_function=cosine_distance, margin=margin, reduction="none"
        )

    def forward(self, features: Tensor, labels: Union[Tensor, List[int]]) -> Tensor:
        labels_list = labels2list(labels)

        anchor, positive, negative = self.miner.sample(
            features=features, labels=labels_list
        )
        loss = self.tri_loss(anchor=anchor, positive=positive, negative=negative)

        self._last_logs.update({"train loss": loss.sum().item()})

        if self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()
        elif self.reduction == "none":
            pass
        else:
            raise ValueError()

        return loss

    @property
    def last_logs(self) -> Dict[str, Any]:
        """
        Returns:
            Dictionary containing useful statistic calculated for the last batch.
        """
        return self._last_logs
