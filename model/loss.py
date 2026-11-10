import ot
import torch
import torch.nn as nn
import torch.nn.functional as F

def find_nearest_neighbors(pred, gt):
    dists = torch.cdist(pred, gt)
    dists_squared = dists ** 2
    
    min_dist_p2g, indices_p2g = dists_squared.min(dim=2)
    min_dist_g2p, _ = dists_squared.min(dim=1)
    
    return min_dist_p2g, indices_p2g, min_dist_g2p


def chamfer_loss(min_dist_p2g, min_dist_g2p):
    return min_dist_p2g.mean() + min_dist_g2p.mean()


def weighted_chamfer_loss(min_dist_p2g, min_dist_g2p, gt_weights):
    term_g2p = (min_dist_g2p * gt_weights).sum() / (gt_weights.sum() + 1e-8)
    term_p2g = min_dist_p2g.mean()
    
    return term_p2g + term_g2p


def sinkhorn_emd_loss(pred_tc, gt_tc, epsilon=0.05, num_itermax=100):
    B, N, _ = pred_tc.shape
    _, M, _ = gt_tc.shape
    
    # Define 1D uniform distributions for the transport plan
    a = torch.ones(N, device=pred_tc.device, dtype=pred_tc.dtype) / N
    b = torch.ones(M, device=gt_tc.device, dtype=gt_tc.dtype) / M

    # Initialize loss accumulator to preserve the computation graph
    # total_sinkhorn_loss = torch.tensor(0.0, device=pred_tc.device, dtype=pred_tc.dtype)
    total_sinkhorn_loss = 0.0

    # Loop over the batch to bypass 3D dimension bugs in POT's PyTorch backend
    for i in range(B):
        p_single = pred_tc[i:i+1]
        g_single = gt_tc[i:i+1]
        
        # Squeeze ensures the cost matrix is strictly 2D [N, M]
        cost_matrix = torch.cdist(p_single, g_single, p=2).squeeze(0) ** 2
        
        # Calculate optimal transport distance
        distance = ot.sinkhorn2(a, b, cost_matrix, reg=epsilon, numItermax=num_itermax)
        total_sinkhorn_loss = total_sinkhorn_loss + distance
        
    return total_sinkhorn_loss / B


def point_to_plane_normal_loss(pred_points, gt_points, gt_normals, indices_p2g):
    B, N, _ = pred_points.shape
    
    batch_idx = torch.arange(B, device=pred_points.device).view(B, 1).expand(-1, N)

    nearest_gt = gt_points[batch_idx, indices_p2g]
    raw_normals = gt_normals[batch_idx, indices_p2g]
    nearest_normal = F.normalize(raw_normals, p=2, dim=-1)

    # Project the displacement vector onto the normal vector
    displacement = pred_points - nearest_gt
    plane_dist_squared = (displacement * nearest_normal).sum(dim=-1).pow(2)

    return plane_dist_squared.mean()


def weighted_point_to_plane_normal_loss(pred_points, gt_points, gt_normals, indices_p2g, gt_weights):
    B, N, _ = pred_points.shape
    batch_idx = torch.arange(B, device=pred_points.device).view(B, 1).expand(-1, N)

    nearest_gt = gt_points[batch_idx, indices_p2g]
    raw_normals = gt_normals[batch_idx, indices_p2g]
    nearest_normal = F.normalize(raw_normals, p=2, dim=-1)
    matched_weights = gt_weights[batch_idx, indices_p2g]

    displacement = pred_points - nearest_gt
    plane_dist_squared = (displacement * nearest_normal).sum(dim=-1).pow(2)
    
    # True normalized weighted average
    weighted_dist = (plane_dist_squared * matched_weights).sum() / (matched_weights.sum() + 1e-8)

    return weighted_dist


def compute_spatial_weights(points, intensity=16.0, power=2.0):
    x_vals = points[..., 0]
    weights = 1.0 + intensity * (x_vals.abs().pow(power))
    return weights.detach()

class LossConfig:

    # Geometric Losses
    CHAMFER_W = 200.0
   
    SPATIAL_INTENSITY = 16.0

    RANGE_W = 10.0

    # Segmentation
    SEG_W_START = 20.0
    SEG_W_END = 2.0
    SEG_DECAY_EPOCHS = 80

    # Delayed High-Frequency Features
    NORMAL_MAX_W = 50.0        
    NORMAL_START_EPOCH = 30
    NORMAL_WARMUP_EPOCHS = 20   

    # Attention Constraints
    VIEW_COVERAGE_W = 10.0
    SPATIAL_DIV_W = 1.0

def range_penalty(p, limit=1.1):
    return torch.relu(p.abs() - limit).pow(2).mean()

def get_segmentation_weight(epoch):
    if epoch >= LossConfig.SEG_DECAY_EPOCHS:
        return LossConfig.SEG_W_END
    t = epoch / LossConfig.SEG_DECAY_EPOCHS
    return LossConfig.SEG_W_START - t * (LossConfig.SEG_W_START - LossConfig.SEG_W_END)

def get_chamfer_weight(epoch):
    return LossConfig.CHAMFER_W

def get_normal_weight(epoch):
    if epoch < LossConfig.NORMAL_START_EPOCH:
        return 0.0
    t = min(1.0, (epoch - LossConfig.NORMAL_START_EPOCH) /
            LossConfig.NORMAL_WARMUP_EPOCHS)
    return LossConfig.NORMAL_MAX_W * t

def view_coverage_loss(view_weights, min_usage=0.05, eps=1e-6):
    # Average across all queries/points to get global view usage: (B, V)
    w = view_weights.mean(dim=1) 
    
    # Normalize to ensure it sums to 1.0
    w = w / (w.sum(dim=-1, keepdim=True) + eps)
    
    # Penalize only if usage drops below the minimum threshold
    return F.relu(min_usage - w).mean()


def spatial_diversity_loss(attn_weights, view_type_flat, view_weights, min_entropy=1.5, eps=1e-6):
    attn_weights = attn_weights.clamp(min=eps)
    B, m, L = attn_weights.shape
    V = view_weights.shape[-1]

    # Initialize as tensor so .sum() works even if all views are skipped
    loss  = attn_weights.new_zeros(B, m)
    count = attn_weights.new_zeros(1)

    for v in range(1, V + 1):
        # 1. Isolate tokens belonging to this specific view
        token_mask = (view_type_flat == v).float()
        if not (token_mask.sum(dim=1) > 0).any():
            continue

        # 2. Extract and re-normalize attention within this view only
        attn_v = attn_weights * token_mask.unsqueeze(1)
        attn_v = attn_v / (attn_v.sum(dim=-1, keepdim=True) + eps)
        
        # 3. Calculate Shannon Entropy along the sequence dimension
        entropy = -(attn_v * torch.log(attn_v + eps)).sum(dim=-1)

        # 4. GATING: Only penalize if the view exists AND the head is actively using it (>5%)
        exists = (token_mask.sum(dim=1) > 0).float()
        gate = (view_weights[..., v - 1] > 0.05).float()
        effective_gate = gate * exists.unsqueeze(1)

        # 5. Apply margin penalty
        penalty = F.relu(min_entropy - entropy)
        
        # Accumulate only the valid, gated penalties
        loss += penalty * effective_gate
        count += effective_gate.sum()

    return loss.sum() / (count + eps)


# Main Loss Moduler
class TotalLoss(nn.Module):
    def __init__(self, exp_config):
        super().__init__()
   
        # Default configuration fallback
        self.config = {
            "recon_type": "emd",              # "chamfer", "weighted_chamfer", "emd"
            "normal_type": "none",            # "none", "normal", "weighted_normal"
            "use_range": False,               # True, False
            "use_seg": True,                  # True, False
            "use_view_coverage": True,        # True, False
            "use_spatial_div": True           # True, False
        }
        self.config.update(exp_config)
        
        self._cached_template_knn = None

    def _generate_seg_mask(self, seg_logits, bboxes, bbox_classes):
        B, n_classes, Hf, Wf = seg_logits.shape
        device = seg_logits.device
        gt_mask = torch.zeros((B, Hf, Wf), dtype=torch.long, device=device)
        for b in range(B):
            valid_indices = torch.nonzero(bbox_classes[b]).squeeze(1)
            for v_idx in valid_indices:
                cls = bbox_classes[b, v_idx].item()
                bx1, by1, bx2, by2 = bboxes[b, v_idx]
                x1 = max(0, int(bx1 * Wf))
                y1 = max(0, int(by1 * Hf))
                x2 = min(Wf, int(torch.ceil(bx2 * Wf)))
                y2 = min(Hf, int(torch.ceil(by2 * Hf)))
                region = gt_mask[b, y1:y2, x1:x2]
                region[region == 0] = cls
        return gt_mask

    def forward(
        self,
        points_pred,
        seg_logits,
        view_weights,
        final_weights,
        view_type_flat,
        points_gt,
        normals_gt,
        bboxes,
        bbox_classes,
        epoch,
    ):
        losses = {}
        device = points_gt.device
        zero = torch.tensor(0.0, device=device)

        # Unpack model predictions
        p_template = points_pred["template"]
        p_refine = points_pred["refined"]
        delta = points_pred["delta"]

        # Dynamic Weights
        w_chamfer = get_chamfer_weight(epoch)
        w_normal = get_normal_weight(epoch)
        w_seg = get_segmentation_weight(epoch)

        losses["meta/w_chamfer"] = torch.tensor(w_chamfer, device=device)
        losses["meta/w_normal"] = torch.tensor(w_normal, device=device)
        losses["meta/w_seg"] = torch.tensor(w_seg, device=device)

        # Base Computations
        needs_weights = (self.config["recon_type"] == "weighted_chamfer" or 
                         self.config["normal_type"] == "weighted_normal")
        if needs_weights:
            gt_weights = compute_spatial_weights(points_gt, intensity=LossConfig.SPATIAL_INTENSITY)
        
        needs_knn = (self.config["recon_type"] in ["chamfer", "weighted_chamfer"] or 
                     self.config["normal_type"] != "none")
        if needs_knn:
            dists_p2g, idx_p2g, dists_g2p = find_nearest_neighbors(p_refine, points_gt)

        # Reconstruction loss
        if self.config["recon_type"] == "emd":
            loss_recon = sinkhorn_emd_loss(p_refine, points_gt)
        elif self.config["recon_type"] == "weighted_chamfer":
            loss_recon = weighted_chamfer_loss(dists_p2g, dists_g2p, gt_weights)
        elif self.config["recon_type"] == "chamfer":
            loss_recon = chamfer_loss(dists_p2g, dists_g2p)
        else:
            raise ValueError(f"Unknown recon_type: {self.config['recon_type']}")

        losses["raw/recon"] = loss_recon
        losses["loss_recon"] = w_chamfer * loss_recon

        # Surface consistency (normal)
        if self.config["normal_type"] != "none" and w_normal > 0.0:
            if self.config["normal_type"] == "weighted_normal":
                loss_normal = weighted_point_to_plane_normal_loss(
                    p_refine, points_gt, normals_gt, idx_p2g, gt_weights)
            elif self.config["normal_type"] == "normal":
                loss_normal = point_to_plane_normal_loss(
                    p_refine, points_gt, normals_gt, idx_p2g)
            
            losses["raw/normal"] = loss_normal
            losses["loss_normal"] = w_normal * loss_normal
        else:
            losses["loss_normal"] = zero

        # Geometry constraint (point cloud range/bounding box)
        if self.config["use_range"]:
            losses["loss_range"] = LossConfig.RANGE_W * range_penalty(p_refine)
        else:
            losses["loss_range"] = zero

        # Segmentation
        if self.config["use_seg"] and w_seg > 0.0:
            gt_seg_mask = self._generate_seg_mask(seg_logits, bboxes, bbox_classes)
            losses["loss_seg"] = w_seg * F.cross_entropy(seg_logits, gt_seg_mask, label_smoothing=0.05)
        else:
            losses["loss_seg"] = zero

        # Attention
        if self.config.get("use_view_coverage",  True):
            losses["loss_view_coverage"] = LossConfig.VIEW_COVERAGE_W * view_coverage_loss(view_weights)
        else:
            losses["loss_view_coverage"] = zero

        if self.config.get("use_spatial_div", True):
            losses["loss_spatial_div"] = LossConfig.SPATIAL_DIV_W * spatial_diversity_loss(final_weights, view_type_flat, view_weights)
        else:
            losses["loss_spatial_div"] = zero

  
        # Total
        total = sum(v for k, v in losses.items() if not k.startswith("meta/") and not k.startswith("raw/"))
        losses["total_loss"] = total

        return losses
