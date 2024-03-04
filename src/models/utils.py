from .base import BaseModel
from .feedforward.model import FeedForward
from .feedforward_prob.model import FeedForwardProb
from .nbeats.model import create_generic_nbeats, create_interpretable_nbeats
from .seq2seq.model import Seq2Seq
from .tcn.model import TCN
from .transformer.model import Transformer


def get_model(model_name: str) -> BaseModel:
    models = {
        "feedforward": FeedForward,
        "feedforwardprob": FeedForwardProb,
        "nbeats_g": create_generic_nbeats,
        "nbeats_i": create_interpretable_nbeats,
        "seq2seq": Seq2Seq,
        "tcn": TCN,
        "transformer": Transformer,
    }
    return models[model_name]
