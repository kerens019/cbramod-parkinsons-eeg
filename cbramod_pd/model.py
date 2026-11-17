"""CBraMod backbone + lightweight prediction head."""

import sys

import torch
import torch.nn as nn

from . import config as C

_WEIGHTS_CACHE = {}


def register_cbramod(cbramod_dir):
    """Make `from cbramod import CBraMod` work.

    `cbramod_dir` is the folder containing cbramod.py, i.e. the `models/`
    folder of https://github.com/wjq-learning/CBraMod.
    """
    if cbramod_dir not in sys.path:
        sys.path.append(cbramod_dir)


class CBraModFineTune(nn.Module):
    """Pretrained CBraMod, mean-pooled to a 200-d embedding, then Dropout + Linear(200 -> 1).

    Input (batch, 19, 30, 200). Output: one value per segment (a logit for
    classification, the predicted score for regression); with
    return_embedding=True also the 200-d segment embedding.
    """

    def __init__(self, weights_path, dropout=C.DROPOUT):
        super().__init__()
        from cbramod import CBraMod

        self.backbone = CBraMod()
        if weights_path not in _WEIGHTS_CACHE:           # read the checkpoint from disk once
            _WEIGHTS_CACHE[weights_path] = torch.load(weights_path, map_location="cpu", weights_only=True)
        self.backbone.load_state_dict(_WEIGHTS_CACHE[weights_path])
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(C.EMBED_DIM, 1))

    def forward(self, x, return_embedding=False):
        feats = self.backbone.encoder(self.backbone.patch_embedding(x))
        emb = feats.mean(dim=(1, 2))                     # mean over channels and patches
        out = self.head(emb).squeeze(1)
        return (out, emb) if return_embedding else out
