import torch
import torch.nn as nn
import torch.nn.functional as F
from model.pos_embeddings import PositionEmbeddingSine2D, PositionEmbeddingSine3D


class PointRefiner(nn.Module):

    def __init__(
        self,
        feat_dim=128,
        hidden_dim=128,
        n_head=4,
        num_layers=4,
        num_views=3,
        temperature=2.0,
        dropout=0.2
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_views = num_views
        self.temperature = temperature

        # Image-side embeddings
        # Project encoder feature map from feat_dim -> hidden_dim
        # Input : (B, feat_dim, Hf, Wf)
        # Output: (B, hidden_dim, Hf, Wf)
        self.feat_proj = nn.Conv2d(feat_dim, hidden_dim, kernel_size=1)

        # Embedding that tells the transformer WHICH view (XY / XZ / YZ)
        # each image token belongs to
        self.view_embedding = nn.Embedding(num_views + 1, hidden_dim)

        #  2D positional encoding for image feature map
        #  Produces sine/cos embeddings aligned with image grid
        self.pos_embed_2d = PositionEmbeddingSine2D(
            num_pos_feats=self.hidden_dim // 2, normalize=True
        )

        # 3D sinusoidal positional encoding for point coordinates
        self.pos_embed_3d = PositionEmbeddingSine3D(
            num_pos_feats=hidden_dim // 2, normalize=False  # Handles scale internally
        )

        # Project 3D positional encoding to transformer dimension
        self.query_pos_proj = nn.Linear((hidden_dim // 2) * 3, hidden_dim)

        # Transformer decoder layers
        self.layers = nn.ModuleList(
            [
                nn.TransformerDecoderLayer(
                    d_model=hidden_dim,
                    nhead=n_head,
                    dim_feedforward=hidden_dim * 4,
                    dropout=dropout,
                    activation=F.silu,
                    batch_first=True,
                )
                for _ in range(num_layers)
            ]
        )

        # Output head: predicts point offsets (dx, dy, dz)
        self.output_head = nn.Sequential(
            # Layer 1
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(inplace=True),
            # Layer 2
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(inplace=True),
            # Layer 3 (Projection)
            nn.Linear(hidden_dim, 3),
        )

        # Learnable scalar
        self.delta_scale = nn.Parameter(torch.tensor(0.01))

        # Init weigths
        self._init_weights()

    def _init_weights(self):
        # Initialize the final projection layer with small noise
        nn.init.normal_(self.output_head[-1].weight, std=1e-4)

        # Force the bias to exactly 0.0
        nn.init.zeros_(self.output_head[-1].bias)

    def forward(self, F1, P, seg_logits, bboxes=None, bbox_mask=None, bbox_classes=None): 
        B, C, Hf, Wf = F1.shape
        device = F1.device
        L_total = Hf * Wf

        # Channel dropout: removes feature channels
        if self.training:
            F1 = F.dropout2d(F1, p=0.1, training=self.training)
       
        # Build view-type map (teacher forcing or prediction)
        if self.training and bboxes is not None and bbox_mask is not None:
            # Teacher forcing: rasterize GT bounding boxes
            view_type_map = torch.zeros(
                (B, Hf, Wf), dtype=torch.long, device=device)

            for b in range(B):
                for v in range(bboxes.shape[1]):

                    if not bbox_mask[b, v]:
                        continue

                    cls = bbox_classes[b, v].item()
                    if cls == 0:
                        continue
                    bx1, by1, bx2, by2 = bboxes[b, v]

                    x1 = max(0, int(bx1 * Wf))
                    y1 = max(0, int(by1 * Hf))
                    x2 = min(Wf, int(bx2 * Wf))
                    y2 = min(Hf, int(by2 * Hf))

                    view_type_map[b, y1:y2, x1:x2] = cls

        else:
            # Inference, use predicted segmentation
            view_type_map = torch.argmax(seg_logits, dim=1)  # (B, Hf, Wf)

        
        # Project image features
        feat = self.feat_proj(F1)

        # Positional encoding for image grid
        pos_enc_map = self.pos_embed_2d(Hf, Wf, device)
        pos_enc_map = pos_enc_map.unsqueeze(0).expand(B, -1, -1, -1)

        # Flatten spatial dimensions
        # L = Hf * Wf (tokens)
        feat_flat = feat.flatten(2).transpose(1, 2)  # (B, L, hidden_dim)
        pos_flat = pos_enc_map.flatten(1, 2)  # (B, L, hidden_dim)
        view_type_flat = view_type_map.flatten(1)  # (B, L)

       
        # Segmentation-Aware Token Selection
        # Calculating base energy (feature magnitude)
        # Shape: (B, L)
        feature_magnitude = feat_flat.abs().max(dim=-1)[0]

        # Calculate structure bias
        if self.training and bboxes is not None:
            # Use the GT view_type_map
            # If it's > 0, we know for a fact it is inside a valid ship view.
            foreground_prob = (view_type_map > 0).float().flatten(1)
        else:
            # Validation/Inference: Rely on the network's own predictions
            with torch.no_grad():
                # Softmax to get probabilities (B, Views+1, Hf, Wf)
                seg_probs = F.softmax(seg_logits, dim=1)

                # Probability that a pixel is NOT background (Class 0)
                # Sum of probs for Classes [1, 2, 3...]
                # Shape: (B, Hf, Wf) -> (B, L)
                foreground_prob = (1.0 - seg_probs[:, 0, :, :]).flatten(1)

        # Compute weighted score (soft gating)
        # Boost the energy of tokens inside the predicted hull.
        # Score = Mag * (1 + beta * Prob)
        bias_strength = 2.0
        token_energy = feature_magnitude * \
            (1.0 + bias_strength * foreground_prob)

        # Select top-K based on new weighted score
        # Keep only the most "active" tokens
        # Since input is inverted (Lines=1), active tokens have high magnitude.
        keep_ratio = 0.20 # TODO: can be hyperparameter
        K = int(L_total * keep_ratio)

        # Find indices of the top-K most important tokens
        # indices: (B, K)
        _, indices = torch.topk(token_energy, k=K, dim=1)

        # Expand indices to match channel dim for gathering
        # (B, K, C)
        indices_expanded = indices.unsqueeze(
            -1).expand(-1, -1, self.hidden_dim)

        # Select features
        feat_selected = torch.gather(feat_flat, 1, indices_expanded)

        # Select positional encodings
        pos_selected = torch.gather(pos_flat, 1, indices_expanded)

        # Select view types
        # view_type_flat is (B, L), gather uses indices (B, K)
        view_type_selected = torch.gather(view_type_flat, 1, indices)

        # Generate view embeddings for only the selected tokens
        view_embs = self.view_embedding(view_type_selected)  # (B, K, C)

        # Final transformer input
        # Keys get the spatial coordinates
        k_input = feat_selected + pos_selected + view_embs

        # Values only get the visual features and the view type
        v_input = feat_selected + view_embs

        # Mask background tokens from attention
        key_padding_mask = view_type_selected == 0

        # Prepare point queries
        # 3D positional encoding for points
        # (B, m, hidden_dim*?)
        raw_pos = self.pos_embed_3d(P)
        query_pos = self.query_pos_proj(raw_pos)  # (B, m, hidden_dim)
        out = torch.zeros_like(query_pos)  # (B, m, hidden_dim)

        # Transformer decoder
        final_weights_compressed = None  # (B, m, K)

        for i, layer in enumerate(self.layers):

            # Self-attention: point ↔ point
            # Q,K = content + 3D position
            # V   = content only
            # Shape:
            #   Q,K,V : (B, m, hidden_dim)
            q = out + query_pos
            tgt2 = layer.self_attn(q, q, out)[0]
            out = layer.norm1(out + layer.dropout1(tgt2))

            # Cross-attention: point ↔ image tokens
            # Query : point (with 3D position)
            # Key   : image tokens (with 2D position + view)
            # Value : image tokens
            scaled_query = (out + query_pos) / self.temperature

            tgt2, weights = layer.multihead_attn(
                query=scaled_query,  # (B, m, hidden_dim)
                key=k_input,  # (B, L, hidden_dim)
                value=v_input,
                key_padding_mask=key_padding_mask,
                need_weights=True,
            )

            if i == self.num_layers - 1:
                final_weights_compressed = weights  # (B, m, L)

            out = layer.norm2(out + layer.dropout2(tgt2))

            # Feed-forward network
            tgt2 = layer.linear2(layer.dropout(
                layer.activation(layer.linear1(out))))
            out = layer.norm3(out + layer.dropout3(tgt2))

        
        # Aggregate spatial attention, per-view attention
        # final_attn: (B, m, L)
        # view_type_flat: (B, L), values in {0, 1..num_views}
        B, m, K_actual = final_weights_compressed.shape
        V = self.num_views

        # Per-point, per-view attention weights
        # Shape: (B, m, V)
        view_weights = final_weights_compressed.new_zeros(B, m, V)

        for v in range(1, V + 1):
            # Mask tokens belonging to view v
            mask = (view_type_selected == v).unsqueeze(1)  # (B, 1, L)
            view_weights[:, :, v -
                         1] = (final_weights_compressed * mask).sum(dim=-1)

        # Create empty canvas (zeros)
        full_attn_map = torch.zeros(
            B, m, L_total, device=device, dtype=final_weights_compressed.dtype
        )

        # Expand indices to match the 'm' dimension: (B, K) -> (B, m, K)
        indices_for_scatter = indices.unsqueeze(1).expand(-1, m, -1)

        # Scatter the compressed weights back to their original positions
        # dim=2 is the 'L' dimension
        full_attn_map.scatter_(2, indices_for_scatter, final_weights_compressed)

        # Predict refined point positions
        raw_delta = self.output_head(out)

        # Apply the learnable scale
        delta = raw_delta * self.delta_scale

        point_cloud = P + delta

        return point_cloud, view_weights, full_attn_map, view_type_flat, delta
