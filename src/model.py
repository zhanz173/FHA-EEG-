import torch
import torch.nn as nn
import numpy as np
from typing import Optional
from dataclasses import dataclass

@dataclass
class NeurologistCorrectionConfig:
    num_neurologists: int
    rater_emb_dim: int = 4
    default_mean_rater: bool = True  # Whether to use the mean rater embedding when neurologist ID is not provided


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

    def forward(self, x, mask=None):
        # x: [B, T, H]
        x_norm = self.norm(x)

        # Step 1: Calculate attention scores (gated mechanism)
        a_v = self.attention_V(x)  # (B, T, hidden_dim)
        a_u = self.attention_U(x)  # (B, T, hidden_dim)

        scores = self.scorer(a_v * a_u) # [B, T, K]
        
        if mask is not None:
            # Mask shape assumption: [B, T]
            scores = scores.masked_fill(mask.unsqueeze(-1) == 0, float('-inf'))
            
        attn_weights = torch.softmax(scores, dim=1) # [B, T, K]
        
        # Efficient Context aggregation using Einstein Summation
        # b: batch, t: time, h: hidden, k: heads
        context = torch.einsum('btk,bth->bkh', attn_weights, x)
        context = self.output_projector(context)
        return context, attn_weights

class RaterCorrectionModule(nn.Module):
    '''
    Simple linear correction model p(y|x, rater) = p(z|x) * p(y|z, rater) = p(z|x) * a + b'''
    def __init__(self, num_neurologists, rater_emb_dim, num_classes):
        assert num_neurologists > 0, "num_neurologists must be > 0 for RaterCorrectionModule"
        assert rater_emb_dim > 0, "rater_emb_dim must be > 0 for RaterCorrectionModule"

        super().__init__()
        self.rater_emb = nn.Embedding(num_neurologists+1, rater_emb_dim) # learning 0 position as mean rater embedding
        self.projector = nn.Sequential(
            nn.LayerNorm(rater_emb_dim),
            nn.Linear(rater_emb_dim, rater_emb_dim),
            nn.GELU(),
            nn.Linear(rater_emb_dim, num_classes * 2) # Predict all class corrections at once [0:mum_classes] for a, [num_classes: 2*num_classes] for b
        )

    def forward(self, z, neurologist_ids):
        # during training, randomly mask out neurologist IDs to learn a "mean rater" embedding in the 0 position for inference when neurologist ID is not available
        if self.training:
            mask = torch.rand(neurologist_ids.shape, device=neurologist_ids.device) < 0.1  # 10% chance to mask
            neurologist_ids = neurologist_ids.masked_fill(mask, 0)  # Masked positions get ID 0 (mean rater)

        z = self.rater_emb(neurologist_ids)
        rater_params = self.projector(z).view(-1, 2, self.num_classes)
        a = rater_params[:, 0, :]
        b = rater_params[:, 1, :]
        rater_delta = a * z + b
        return rater_delta

class Identity(nn.Module):
    def forward(self, x, *ignore): 
        return x

class GRU_Classifier(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_layers: int,
        num_classes: int = 5,
        num_pool_heads: int = 4,
        neurologist_correction_config: Optional[NeurologistCorrectionConfig] = None,
        use_transformer: bool = False, # Architectural toggle
        pooling_output_size: Optional[int] = None
        
    ):
        super().__init__()
        self.neurologist_correction_config = neurologist_correction_config
        self.num_classes = num_classes

        # --- Input Normalization ---
        self.LBR_norm = nn.LayerNorm(200)
        self.Welch_norm = nn.LayerNorm(50)

        # --- Encoders ---
        # Welch: Optimized with standard stride patterns
        self.F_encoder = nn.Sequential(
            nn.Conv2d(20, 32, 5, padding=2), nn.BatchNorm2d(32), nn.GELU(),
            nn.MaxPool2d((2,1), stride=(2,1)),
            nn.Dropout(0.1),
            nn.Conv2d(32, 32, 5, padding=2), nn.BatchNorm2d(32), nn.GELU(),
            nn.MaxPool2d((2,1), stride=(2,1)),
            nn.Dropout(0.1),
            nn.Conv2d(32, 64, 5, padding=2), nn.BatchNorm2d(64), nn.GELU(),
            nn.Dropout(0.1),
            nn.AdaptiveMaxPool2d((1, None)) # Output: [B, 64, 1, T]
        )
        self.f_dim = 64 # Output dim of F_encoder

        # LaBraM:
        self.L_encoder = nn.Sequential(
            nn.Linear(200, hidden_size), nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, hidden_size), nn.GELU(),
            nn.Dropout(0.1)
        )

        # --- Gated Fusion ---
        self.fusion = GatedFusion(hidden_size, self.f_dim, hidden_size)

        # --- Temporal Backbone ---
        if use_transformer:
            encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_size, nhead=4, batch_first=True)
            self.backbone = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        else:
            self.backbone = nn.GRU(hidden_size, hidden_size, num_layers, 
                                   batch_first=True, dropout=0.1 if num_layers > 1 else 0)
        self.dropout = nn.Dropout(0.1)


        # --- Pooling & Heads ---
        self.shared_pool = EfficientAttnPool(hidden_size, output_size=pooling_output_size or hidden_size, num_heads=num_pool_heads)
        # Vectorized Head 
        top_in_dim = num_pool_heads * (pooling_output_size or hidden_size)

        self.pred_head = VectorizedPredHead(top_in_dim, num_classes)

        # --- Rater Correction ---
        if neurologist_correction_config is not None:
            self.rater_correction = RaterCorrectionModule(
                num_neurologists=neurologist_correction_config.num_neurologists,
                rater_emb_dim=neurologist_correction_config.rater_emb_dim,
                num_classes=num_classes)
        else:
            self.rater_correction = Identity()  # No correction applied


    def encode_features(self, x_lbr, x_welch):
        # x_lbr: [B, T, 200]
        # x_welch: [B, T, 20, F] -> Permute to [B, 20, F, T] for Conv2d
        
        # Welch path
        x_welch = torch.log1p(x_welch) # Log scaling
        x_w = self.Welch_norm(x_welch) 
        x_w = x_w.permute(0, 2, 3, 1) 
        f_feat = self.F_encoder(x_w) # [B, 64, 1, T]
        f_feat = f_feat.squeeze(2).permute(0, 2, 1) # [B, T, 64]

        # LaBraM path
        x_l = self.LBR_norm(x_lbr)
        l_feat = self.L_encoder(x_l) # [B, T, H]

        # Fused path
        return self.fusion(l_feat, f_feat)
    
    def extract_embeddings(self, x, mask=None):
        x_lbr, x_welch = x
        x_fused = self.encode_features(x_lbr, x_welch) # [B, T, H]
        out = self.backbone(x_fused)

        if isinstance(out, tuple): out = out[0] # Handle GRU output tuple
        ctx, attn_weights = self.shared_pool(out)
        ctx = ctx.detach().mean(dim=1)  # Mean over time
        return ctx, attn_weights  # [B, H]
    
    def get_aux_loss(self, **targets): 
        '''
        In case we want to add auxiliary losses (e.g., age regression), we can return the predicted age here for MSE loss calculation.

        default return zero tensor to avoid breaking existing training loop if not used.
        '''
        return torch.tensor(0.0)

    def forward(self, x, mask=None, neurologist_ids=None):
        x_lbr, x_welch = x
        
        # 1. Fuse
        x_fused = self.encode_features(x_lbr.contiguous(), x_welch.contiguous()) # [B, T, H]
        
        # 2. Backbone (GRU or Transformer)
        out = self.backbone(x_fused)
        if isinstance(out, tuple): out = out[0] # Handle GRU output tuple
        out = self.dropout(out)

        # 3. Pool
        # ctx: [B, K, H], weights: [B, T, K] (useful for visualization)
        ctx, attn_weights = self.shared_pool(out, mask=mask)
        flat = ctx.flatten(start_dim=1) # [B, K*H]

        # 4. Predict (Vectorized)
        logits = self.pred_head(flat) # [B, num_classes]

        # 5. Correction (if neurologist IDs provided)
        logits = self.rater_correction(logits, neurologist_ids) if neurologist_ids is not None else logits

        return logits, attn_weights


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# expect data shape: tuple(LaBraM shape: (89, 200), Welch shape: (89, 20, 25)), Label: [1. 1. 1. 1. 1.]
def unit_test():
    model = GRU_Classifier(hidden_size=64, num_layers=3, num_classes=5, num_pool_heads=3, neurologist_correction_config=NeurologistCorrectionConfig(num_neurologists=10, rater_emb_dim=8), use_transformer=False, pooling_output_size=64)
    print("Number of trainable parameters:", count_parameters(model))
    x_lbr = torch.randn(2, 89, 200)  # batch of 4, 89 time steps, 200 LaBraM features
    x_welch = torch.abs(torch.randn(2, 89, 20, 50))  # batch of 4, 89 time steps, 20 channels, 25 Welch features
    outputs, _ = model((x_lbr, x_welch))
    print("Output:", outputs)  # Expected: (4, 5)
    assert outputs.shape == (2, 5), "Output shape is incorrect"

if __name__ == "__main__":
    unit_test()
