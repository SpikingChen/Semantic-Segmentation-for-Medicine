import torch
from torch import Tensor, einsum
import numpy as np


def probs2class(probs: Tensor) -> Tensor:
    b, _, w, h = probs.shape  # type: Tuple[int, int, int, int]
    assert simplex(probs)

    res = probs.argmax(dim=1)
    assert res.shape == (b, w, h)

    return res


def probs2one_hot(probs: Tensor) -> Tensor:
    _, C, _, _ = probs.shape
    assert simplex(probs)

    res = class2one_hot(probs2class(probs), C)
    assert res.shape == probs.shape
    assert one_hot(res)

    return res


def class2one_hot(seg: Tensor, C: int) -> Tensor:
    n, b, w, h = seg.shape  # type: Tuple[int, int, int]

    res = torch.stack([seg == c for c in range(C)], dim=1).type(torch.int32)

    return res


from scipy.ndimage import distance_transform_edt as distance


def one_hot2dist(seg: np.ndarray) -> np.ndarray:
    # assert one_hot(torch.Tensor(seg), axis=0)
    seg = np.array(seg.cpu())
    N: int = seg.shape[0]
    C: int = seg.shape[1]

    res = np.zeros_like(seg)
    for n in range(N):
        for c in range(C):
            posmask = seg[n][c].astype(np.bool)

            if posmask.any():
                negmask = ~posmask
                # print('negmask:', negmask)
                # print('distance(negmask):', distance(negmask))
                res[n][c] = distance(negmask) * negmask - (distance(posmask) - 1) * posmask
                # print('res[c]', res[c])
    return res


def simplex(t: Tensor, axis=1) -> bool:
    _sum = t.sum(axis).type(torch.float32)
    _ones = torch.ones_like(_sum, dtype=torch.float32)
    return torch.allclose(_sum, _ones)


def one_hot(t: Tensor, axis=1) -> bool:
    return simplex(t, axis) and sset(t, [0, 1])

    # Assert utils


def uniq(a: Tensor):
    return set(torch.unique(a.cpu()).numpy())


def sset(a: Tensor, sub) -> bool:
    return uniq(a).issubset(sub)


class SurfaceLoss:
    def __init__(self, **kwargs):
        # Self.idc is used to filter out some classes of the target mask. Use fancy indexing
        self.idc = kwargs["idc"]
        print(f"Initialized {self.__class__.__name__} with {kwargs}")

    def __call__(self, probs, dist_maps):
        pc = probs[:, self.idc, ...].type(torch.float32)
        dc = dist_maps[:, self.idc, ...].type(torch.float32)

        multipled = einsum("bkwh,bkwh->bkwh", pc, dc)

        loss = multipled.mean()

        return loss


BoundaryLoss = SurfaceLoss
