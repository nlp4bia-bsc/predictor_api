from transformers import RobertaConfig

"""
This config file contains the parameters of the best trained model. They must be copied exactly in order to build the exact architecture the weights will subsequently fill.
"""

class LSTMBERTConfig(RobertaConfig):
    model_type = "lstm-attn-bert"

    def __init__(self, **kwargs):
        # Let RobertaConfig initialize all standard attributes
        super().__init__(**kwargs)

        # Store any extra keys as attributes without listing them
        for k, v in kwargs.items():
            if not hasattr(self, k):
                setattr(self, k, v)