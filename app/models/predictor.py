from transformers import RobertaForSequenceClassification, RobertaModel, RobertaTokenizer
from transformers.modeling_outputs import SequenceClassifierOutput
from torch import nn
import torch
from typing import Optional
from datetime import datetime

from app.models.base_model import ModelClass
from app.config.model_config import LSTMBERTConfig
from app.models.utils import date_linear_impute, dates_to_log_deltas

class LSTMBERT(RobertaForSequenceClassification):
    # IMPORTANT: match HF expected signature: only `config` (other kwargs allowed)
    def __init__(self, config: LSTMBERTConfig, **kwargs):
        super().__init__(config)
        self.config = config

        # Build backbone *without* loading weights here.
        # (RobertaModel(config) creates the module structure that from_pretrained will fill.)
        self.roberta = RobertaModel(config)

        lstm_input_dim = self.roberta.config.hidden_size + config.visit_time_dim

        self.lstm = nn.LSTM(
            lstm_input_dim,
            getattr(config, "lstm_hidden"),
            batch_first=True,
            bidirectional=True,
            num_layers=getattr(config, "lstm_layers", 1),
        )

        attn_dim = getattr(config, "attn_dim")
        self.attn = nn.Sequential(
            nn.Linear(config.lstm_hidden * 2, attn_dim),
            nn.Tanh(),
            nn.Linear(attn_dim, 1)
        )

        self.classifier = nn.Linear(config.lstm_hidden * 2, config.output_dim)

    def forward(self, *args, visit_times: torch.Tensor, **kwargs) -> SequenceClassifierOutput:
        """
        kwargs:
            input_ids: (V, S) long
            attention_mask: (V, S) long
        visit_times: tensor (V, visit_time_dim) float
        
        where V = number of visits in batch, S = max seq len per visit
        """
        input_ids = kwargs.get("input_ids", args[0] if len(args) > 0 else None)
        attention_mask = kwargs.get("attention_mask", args[1] if len(args) > 1 else None)
        if input_ids is None or attention_mask is None:
            raise ValueError("You have to specify input_ids and attention_mask")
        V, S = input_ids.shape
        
        # Check visit_times shape if needed
        if visit_times is None:
            raise ValueError("You have to provide visit_times tensor")
        if visit_times.shape != (V, self.config.visit_time_dim):
            raise ValueError(f"visit_times shape must be (V, {self.config.visit_time_dim})")
        
        # Process each visit through RoBERTa
        pooled_visits = []
        for i in range(V):
            out = self.roberta(
                input_ids=input_ids[i:i+1],  # (1, S)
                attention_mask=attention_mask[i:i+1],
                return_dict=True
            )
            last_hidden = out.last_hidden_state  # (1, S, hidden)
            mask = attention_mask[i:i+1].unsqueeze(-1)  # (1, S, 1)
            cls_vec = (last_hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)  # (1, hidden)
            pooled_visits.append(cls_vec)
        
        proj = torch.cat(pooled_visits, dim=0)  # (V, hidden)
        
        # Concatenate visit times if provided
        proj = torch.cat([proj, visit_times], dim=-1)  # (V, hidden + visit_time_dim)
        
        # Add batch dimension for LSTM
        proj = proj.unsqueeze(0)  # (1, V, hidden + visit_time_dim)
        
        # Run LSTM
        lstm_out, _ = self.lstm(proj)  # (1, V, 2*lstm_hidden)
        lstm_out = lstm_out.squeeze(0)  # (V, 2*lstm_hidden)
        
        # Attention pooling over visits
        scores = self.attn(lstm_out).squeeze(-1)  # (V,)
        attention_weights = torch.softmax(scores, dim=0)  # (V,)
        
        # Weighted sum
        pooled = (attention_weights.unsqueeze(-1) * lstm_out).sum(dim=0)  # (2*lstm_hidden,)
        
        # Classification
        logits = self.classifier(pooled)  # (output_dim,)
        
        return SequenceClassifierOutput(
            logits=logits,
            attentions=attention_weights.detach().cpu().numpy().tolist()
        )
    
class PredictionPipeline(ModelClass):
    def __init__(
            self,
            local_model_path: str
        ):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.cfg = LSTMBERTConfig.from_pretrained(local_model_path)

        self.tokenizer = RobertaTokenizer.from_pretrained(
            local_model_path,
            local_files_only=True
        )
        
        self.model = LSTMBERT.from_pretrained(local_model_path, config=self.cfg, local_files_only=True)
        self.model.eval()
        self.model.to(self.device) # type: ignore

    def predict(
            self,
            case: list[str],
            visit_times: list[str]
        ) -> tuple[float, list[float]]:

        inputs = self.tokenizer(case, return_tensors='pt', max_length=self.cfg.max_length, truncation=True, padding='max_length')
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        visit_times_list = self.format_dates(visit_times)    
        visit_times_tensor = torch.tensor(visit_times_list, dtype=torch.float32).to(self.device)
        outputs = self.model(inputs['input_ids'], inputs['attention_mask'], visit_times=visit_times_tensor)
        syn_prob = outputs.logits.softmax(dim=-1)[1].item()
        attn_list = outputs.attentions
        return syn_prob, attn_list
    
    @staticmethod
    def format_dates(visit_dates_str: list[str]) -> list[tuple[float, float]]:
        """
        Convert one case's ordered string dates dates into two differnce arrays:
        - log_prev: log1p(delta since previous visit)  (first visit -> 0)
        - log_start: log1p(delta since first visit)    (first visit -> 0)

        Inputs:
        list of string dates in this format: [10Jan2024, 9Apr2024, ...]
        Returns:
        list of tuples [(log_prev0, log_start0), ...] length == len(visit_dates_str)
        """
        visit_dates = [datetime.strptime(d, "%d%b%Y") if d else None for d in visit_dates_str]
        visit_dates_imp = date_linear_impute(visit_dates)
        return dates_to_log_deltas(visit_dates_imp)
        