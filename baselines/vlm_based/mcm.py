from baselines.base import BaseBaseline
from baselines.registry import register_baseline
import torch


@register_baseline("MCM")
class MCM(BaseBaseline):
    def score():
        # 这里只写接口示意 具体实现你自己填
        # if logits is None:
        #     raise ValueError("MSPBaseline needs logits")
        # with torch.no_grad():
        #     probs = logits.softmax(dim=1)
        #     scores = probs.max(dim=1).values
        # return scores
        print("mcm")
