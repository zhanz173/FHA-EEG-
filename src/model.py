import torch
import torch.nn as nn
import numpy as np
from typing import Optional
from dataclasses import dataclass
from src.components import EEGGATEncoder, EEGCONVEncoder, build_eeg_edge_index, EEG_GRAPH_NEIGHBORS, EEG_CHANNELS_ORDER

@dataclass
class NeurologistCorrectionConfig:
    num_neurologists: int
    rater_emb_dim: int = 4
    default_mean_rater: bool = True  # UNUSED: Whether to use the mean rater embedding when neurologist ID is not provided


class VectorizedPredHead(nn.Module):
    """
    Optimized CORN head that computes all classes and ranks in a single matrix multiplication.
    Input: [B, in_dim]
    Output: [B, num_classes, ]
    """
    def __init__(self, in_dim, num_classes):
        super().__init__()
        self.num_classes = num_classes
        self.out_dim = num_classes

        self.mlp = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, in_dim),
            nn.GELU(),
            # Predict all (Class x Ranks) at once
            nn.Linear(in_dim, self.out_dim)
        )

    def forward(self, x):
        return self.mlp(x)

class GatedFusion(nn.Module):
    """
    Gated fusion to dynamically weight feature importance before merging.
    """
    def __init__(self, dim1, dim2, out_dim):
        super().__init__()
        self.project_1 = nn.Linear(dim1, out_dim, bias=False)
        self.project_2 = nn.Linear(dim2, out_dim, bias=False)
        self.gate = nn.Linear(dim1 + dim2, out_dim)
        self.norm = nn.LayerNorm(out_dim)
        self.act = nn.GELU()

    def forward(self, x1, x2):
        # x1: [B, T, D1], x2: [B, T, D2]
        h1 = self.project_1(x1)
        h2 = self.project_2(x2)
        
        # Calculate gate based on both inputs
        cat = torch.cat([x1, x2], dim=-1)
        z = torch.sigmoid(self.gate(cat))
        
        # Weighted sum based on gate
        out = z * h1 + (1 - z) * h2
        return self.norm(self.act(out))

class EfficientAttnPool(nn.Module):
    def __init__(self, hidden_size, output_size, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.norm = nn.LayerNorm(hidden_size)

        self.attention_V = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh()
        )
        self.attention_U = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Sigmoid()
        )
        
        # Combine w0 and w into one sequential block for cleaner gradients
        self.scorer = nn.Sequential(
            nn.Linear(hidden_size, num_heads)
        )
        if hidden_size != output_size:
            self.output_projector = nn.Linear(hidden_size, output_size)
        else:
            self.output_projector = nn.Identity()

    def forward(self, x, mask=None): # [B, T, H]-> [B, num_heads, output_size]
        # x: [B, T, H]
        x = self.norm(x)

        # Step 1: Calculate attention scores (gated mechanism)
        a_v = self.attention_V(x)  # (B, T, hidden_dim)
        a_u = self.attention_U(x)  # (B, T, hidden_dim)

        scores = self.scorer(a_v * a_u) # [B, T, K]
        
        if mask is not None:
            # Mask shape assumption: [B, T]
            scores = scores.masked_fill(mask.unsqueeze(-1) == 0, -1e9)
            
        attn_weights = torch.softmax(scores, dim=1) # [B, T, K]
        
        # Efficient Context aggregation using Einstein Summation
        # b: batch, t: time, h: hidden, k: heads
        context = torch.einsum('btk,bth->bkh', attn_weights, x)
        context = self.output_projector(context)
        return context, attn_weights
    
class RaterCorrectionModule(nn.Module):
    def __init__(self, num_neurologists, rater_emb_dim, num_classes):
        super().__init__()
        self.num_classes = num_classes
        self.rater_emb = nn.Embedding(num_neurologists + 1, rater_emb_dim)
        self.projector = nn.Sequential(
            nn.LayerNorm(rater_emb_dim),
            nn.Linear(rater_emb_dim, rater_emb_dim),
            nn.GELU(),
            nn.Linear(rater_emb_dim, num_classes * 2)
        )

        # initialize final layer near zero => near identity correction
        nn.init.zeros_(self.projector[-1].weight)
        nn.init.zeros_(self.projector[-1].bias)

    def forward(self, z, neurologist_ids):
        z_h = self.rater_emb(neurologist_ids)
        params = self.projector(z_h).view(-1, 2, self.num_classes)
        delta_a = params[:, 0, :]
        delta_b = params[:, 1, :]

        scale = 1.0 + 0.1 * delta_a   # optionally damp it
        bias  = 0.1 * delta_b
        logits_post = scale * z + bias

        reg = (delta_a ** 2 + delta_b ** 2).mean()
        return logits_post, reg

class Identity(nn.Module):
    def forward(self, x, *ignore): 
        return x
    
class MeanPool(nn.Module):
    def forward(self, x, mask=None):
        if mask is not None:
            mask = mask.unsqueeze(-1)  # [B, T, 1]
            x = x * mask  # Zero out masked positions
            sum_x = x.sum(dim=1)  # Sum over time
            count = mask.sum(dim=1)  # Count of valid (non-masked) positions
            mean_x = sum_x / (count + 1e-8)  # Avoid division by zero
        else:
            mean_x = x.mean(dim=1)
        return mean_x, None

class GRU_Classifier(nn.Module):
    def __init__(
        self,
        encoder_hidden_size: int = 64,
        RNN_hidden_size: int = 64,
        num_layers: int = 2,
        num_classes: int = 5,
        num_pool_heads: int = 4,
        neurologist_correction_config: Optional[NeurologistCorrectionConfig] = None,
        use_transformer: bool = False, # Architectural toggle
        pooling_output_size: Optional[int] = None,
        use_mean_pooling: bool = False
    ):
        super().__init__()
        self.neurologist_correction_config = neurologist_correction_config
        self.num_classes = num_classes

        # --- Input Normalization ---
        self.LBR_norm = nn.LayerNorm(200)
        self.Welch_norm = nn.LayerNorm(50)

        # --- Encoders ---
        # Welch: Optimized with standard stride patterns
        self.F_encoder = EEGGATEncoder(in_features=50, hidden_features=encoder_hidden_size, out_features=encoder_hidden_size, pool='mean', dropout=0.1)

        # LaBraM:
        self.L_encoder = nn.Sequential(
            nn.Linear(200, encoder_hidden_size), nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(encoder_hidden_size, encoder_hidden_size), nn.GELU(),
            nn.Dropout(0.1)
        )

        # --- Gated Fusion ---
        self.fusion = GatedFusion(encoder_hidden_size, encoder_hidden_size, RNN_hidden_size)

        # --- Temporal Backbone ---
        self.backbone = nn.GRU(RNN_hidden_size, RNN_hidden_size, num_layers, 
                                   batch_first=True, dropout=0.1 if num_layers > 1 else 0)
        self.dropout = nn.Dropout(0.1)


        # --- Pooling & Heads ---
        if use_mean_pooling:
            self.shared_pool = MeanPool()  # Just averages over time steps, no attention
            top_in_dim = RNN_hidden_size
        else:
            self.shared_pool = EfficientAttnPool(RNN_hidden_size, output_size=pooling_output_size or RNN_hidden_size, num_heads=num_pool_heads)
            top_in_dim = num_pool_heads * (pooling_output_size or RNN_hidden_size)
        # Vectorized Head 

        self.pred_head = VectorizedPredHead(top_in_dim, num_classes)

        # --- Rater Correction ---
        if neurologist_correction_config is not None:
            self.rater_correction = RaterCorrectionModule(
                num_neurologists=neurologist_correction_config.num_neurologists,
                rater_emb_dim=neurologist_correction_config.rater_emb_dim,
                num_classes=num_classes)
        else:
            self.rater_correction = None # No correction applied

        edge_index = build_eeg_edge_index(EEG_CHANNELS_ORDER, EEG_GRAPH_NEIGHBORS)
        self.register_buffer("edge_index", edge_index)

    def encode_features(self, x_lbr, x_welch, mask=None):
        x_welch = torch.log1p(x_welch)
        x_w = self.Welch_norm(x_welch)
        x_w = x_w.permute(0, 3, 2, 1)
        f_feat = self.F_encoder(x_w, edge_index=self.edge_index)

        x_l = self.LBR_norm(x_lbr)
        l_feat = self.L_encoder(x_l)

        if mask is not None:
            m = mask.unsqueeze(-1).to(l_feat.dtype)
            l_feat = l_feat * m
            f_feat = f_feat * m

        fused = self.fusion(l_feat, f_feat)

        if mask is not None:
            fused = fused * m

        return fused
    
    def extract_embeddings(self, x, mean_over_time=False):
        x_lbr, x_welch = x
        x_fused = self.encode_features(x_lbr, x_welch) # [B, T, H]
        out = self.backbone(x_fused)

        if isinstance(out, tuple): out = out[0] # Handle GRU output tuple
        ctx, attn_weights = self.shared_pool(out)
        if mean_over_time:
            out = out.detach().mean(dim=1)  # Mean over time
        return out, attn_weights  # [B, H]
    
    def get_aux_loss(self, **targets): 
        '''
        In case we want to add auxiliary losses (e.g., age regression), we can return the predicted age here for MSE loss calculation.

        default return zero tensor to avoid breaking existing training loop if not used.
        '''
        
        return 0.01* self.l2_loss
    
    def load_pretrained_encoder_weights(self, pretrained_state_dict):
        # Assuming the state dict is for the entire model, we need to extract the encoder part
        encoder_state_dict = {k.replace("F_encoder.", ""): v for k, v in pretrained_state_dict.items() if k.startswith("F_encoder.")}
        self.F_encoder.load_state_dict(encoder_state_dict, strict=False)  # Load with strict=False to ignore missing keys
        print("Pretrained encoder loaded successfully.")

    def forward(self, x, mask=None, lengths=None, neurologist_ids=None):
        x_lbr, x_welch = x
        x_fused = self.encode_features(x_lbr.contiguous(), x_welch.contiguous(), mask=mask)

        if lengths is not None:
            packed = nn.utils.rnn.pack_padded_sequence(
                x_fused,
                lengths.cpu(),
                batch_first=True,
                enforce_sorted=False
            )
            out, _ = self.backbone(packed)
            out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True)
        else:
            out = self.backbone(x_fused)
            if isinstance(out, tuple):
                out = out[0]

        if isinstance(out, tuple):
            out = out[0]

        out = self.dropout(out)
        ctx, attn_weights = self.shared_pool(out, mask=mask)
        flat = ctx.flatten(start_dim=1)
        logits = self.pred_head(flat)

        if neurologist_ids is not None and self.rater_correction is not None:
            logits_post, l2_loss = self.rater_correction(logits, neurologist_ids)
        else:
            logits_post, l2_loss = logits, 0

        self.l2_loss = l2_loss
        return logits_post, attn_weights

    
def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# expect data shape: tuple(LaBraM shape: (89, 200), Welch shape: (89, 20, 25)), Label: [1. 1. 1. 1. 1.]
def unit_test():
    model = GRU_Classifier(encoder_hidden_size=128, RNN_hidden_size=64, num_layers=2, num_classes=5, num_pool_heads=3, neurologist_correction_config=NeurologistCorrectionConfig(num_neurologists=10, rater_emb_dim=8), use_mean_pooling=True, pooling_output_size=64)
    print("Number of trainable parameters:", count_parameters(model))
    x_lbr = torch.randn(2, 89, 200)  # batch of 2, 89 time steps, 200 LaBraM features
    x_welch = torch.abs(torch.randn(2, 89, 20, 50))  # batch of 2, 89 time steps, 20 channels, 50 Welch features
    outputs, _ = model((x_lbr, x_welch), neurologist_ids=torch.tensor([1, 2]))  # Provide neurologist IDs for correction
    print("Output:", outputs)  # Expected: (4, 5)
    assert outputs.shape == (2, 5), "Output shape is incorrect"
    model = model.cuda()
    print(model.edge_index.device)

if __name__ == "__main__":
    unit_test()
