import torch
import numpy as np

def apply_ancestral_resonance(preds, ancestor_matrix_sparse, alpha=0.5, device='cuda'):
    """
    Applies Ancestral Resonance to smooth predictions using graph topology.

    Principle:
        Final_Score = (Self_Score ^ alpha) * (Ancestral_Support ^ (1-alpha))
        Using Geometric Mean approach to act as a soft AND gate.

    Args:
        preds (torch.Tensor): (N, 40000) Raw Probabilities [0.0 ~ 1.0]
        ancestor_matrix_sparse (scipy.sparse.csr_matrix): (40000, 40000)
            - row i, col j = 1 means term j is an ancestor of term i.
        alpha (float): Weight for self-score (0.0 ~ 1.0).
            - 0.5: Geometric Mean (Balanced)
            - 0.8: Trust Self more
            - 0.2: Trust Ancestors more

    Returns:
        torch.Tensor: Resonated probabilities.
    """
    # 1. Ensure Inputs are on GPU
    if not isinstance(preds, torch.Tensor):
        preds = torch.tensor(preds, device=device, dtype=torch.float32)

    # 2. Convert Scipy Sparse Matrix to Torch Sparse Tensor (Efficiently)
    # This should ideally be done once and cached, but doing it here for completeness.
    if not isinstance(ancestor_matrix_sparse, torch.Tensor):
        # Ensure it's COO format for easy conversion
        coo = ancestor_matrix_sparse.tocoo()

        indices = torch.LongTensor(np.vstack((coo.row, coo.col))).to(device)
        values = torch.FloatTensor(coo.data).to(device)
        shape = coo.shape

        # Adjacency Matrix (Terms x Terms)
        # Transpose is needed because we want: For term i, get sum of ancestors j
        # Original: A[i, j] = 1 if j is ancestor of i.
        # We need to aggregate columns j for each row i.
        adj = torch.sparse_coo_tensor(indices, values, torch.Size(shape), device=device)
    else:
        adj = ancestor_matrix_sparse

    # 3. Calculate Ancestral Support (Mean Score of Lineage)
    # Operation: Preds (N, L) @ Adj.T (L, L) -> Sum of ancestor scores for each term
    # Note: Sparse MM requires (Sparse, Dense), so we do (Adj @ Preds.T).T
    # But adjacency is L x L, Preds is N x L.
    # We want for each protein (row), for each term (col), the average of its ancestors.

    # Correct Operation:
    # Preds (N, L) * Adj.T (L, L) = (N, L)
    # Logic: For term 't', Sum(Score(ancestor))

    # Transpose Adj for matrix multiplication alignment
    # If Adj[i, j] = 1 (j is ancestor of i)
    # We want result[p, i] = Sum_j (preds[p, j] * Adj[i, j])
    # This is exactly (preds @ Adj.T)

    # Efficient Sparse Logic:
    # Use torch.sparse.mm with transposed adjacency
    # Note: torch.sparse.mm supports (Sparse, Dense) -> Dense
    # We need (Dense @ Sparse). This is equivalent to (Sparse.T @ Dense.T).T

    # Step A: Ancestor Sum
    # (L, L) sparse @ (L, N) dense -> (L, N) dense -> Transpose -> (N, L)
    ancestor_sum = torch.sparse.mm(adj, preds.T).T

    # Step B: Ancestor Count
    # Calculate how many ancestors each term has to compute mean
    # (L, L) sparse @ (L, 1) ones -> (L, 1)
    ones_vec = torch.ones(adj.shape[1], 1, device=device)
    ancestor_counts = torch.sparse.mm(adj, ones_vec).T # (1, L)

    # Avoid division by zero (Root terms might have 0 ancestors in matrix depending on definition)
    ancestor_counts = torch.clamp(ancestor_counts, min=1.0)

    # Step C: Ancestral Support (Average)
    ancestor_support = ancestor_sum / ancestor_counts

    # 4. Resonance Fusion (Geometric Mean)
    # The Genius Step: Soft AND Gate
    # resonated = self^alpha * support^(1-alpha)

    # Add epsilon for log stability if needed, but direct power is fine for [0,1]
    resonated_preds = torch.pow(preds, alpha) * torch.pow(ancestor_support, (1.0 - alpha))

    return resonated_preds

# ==============================================================================
# 🚀 통합 사용 가이드 (Integration Guide)
# ==============================================================================
# 이 함수는 validation.py 또는 inference.py의 마지막 단계에서 호출하십시오.

def run_post_processing_pipeline(preds, ancestor_matrix, ic_weights, device='cuda'):
    print("🔬 Applying Ancestral Resonance (The Genius Smoothing)...")

    # 1. 조상 공명 적용 (Boosting & Denoising)
    # alpha=0.5 -> 나와 족보를 50:50으로 신뢰 (Geometric Mean)
    resonated = apply_ancestral_resonance(preds, ancestor_matrix, alpha=0.5, device=device)

    print("✂️ Applying IC-Adaptive Pruning (The Risk Manager)...")

    # 2. 리스크 기반 컷팅 (Final Decision)
    # 이미 공명으로 노이즈가 약해졌으므로, Base Threshold를 낮게 잡아도 안전함
    # Base=0.25: 공명된 신호가 0.25만 넘으면 합격 (High Recall)
    # Alpha=0.03: 희귀 기능은 엄격하게 (High Precision)

    ic_weights_tensor = torch.tensor(ic_weights, device=device, dtype=torch.float32).unsqueeze(0)
    dynamic_thr = 0.25 + (0.03 * ic_weights_tensor)

    mask = resonated >= dynamic_thr
    final_preds = resonated * mask.float()

    return final_preds
