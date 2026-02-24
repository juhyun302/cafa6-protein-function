"""
Attention-based Multiple Instance Learning (MIL) Model
======================================================
단백질 기능 예측을 위한 Attention-based MIL 모델

입력: (batch, max_chunks, embedding_dim) - 가장 긴 chunk 길이에 맞추고 빈 곳은 zero padding
출력: (batch, num_classes)

Architecture:
1. Chunk Embedding Processing
2. Attention Aggregation (Gated Attention or Standard Attention)
3. Classification Head

사용법:
    from attention_mil_model import AttentionMILClassifier, MILModelConfig
    
    # 방법 1: MILModelConfig 사용
    config = MILModelConfig(
        embedding_dim=2560,
        num_classes=40122,
        mil_hidden_dim=512,
        use_gated_attention=True,
        dropout=0.3
    )
    model = AttentionMILClassifier.from_config(config)
    
    # 방법 2: 직접 파라미터 전달
    model = AttentionMILClassifier(
        embedding_dim=2560,
        num_classes=40122,
        mil_hidden_dim=512,
        use_gated_attention=True,
        dropout=0.3
    )
"""

import torch
import torch.nn as nn
from dataclasses import dataclass, field
from typing import Optional, Dict, Any


@dataclass
class MILModelConfig:
    """
    Attention-based MIL 모델 설정 클래스
    
    Attributes:
        embedding_dim: 입력 임베딩 차원 (ESM2-3B: 2560)
        num_classes: 출력 클래스 수 (GO Term 수)
        mil_hidden_dim: Attention 망과 분류기의 내부 차원
        use_gated_attention: True면 성능이 더 좋은 Gated Attention 사용
        dropout: Dropout 비율 (0.0 ~ 1.0)
    """
    # 필수 파라미터
    embedding_dim: int = 2560
    num_classes: int = 40122
    
    # MIL 구조 설정
    mil_hidden_dim: int = 512
    use_gated_attention: bool = True
    
    # Regularization 설정
    dropout: float = 0.3
    
    def __post_init__(self):
        """설정 유효성 검사"""
        if self.dropout < 0 or self.dropout > 1:
            raise ValueError(f"dropout은 0과 1 사이여야 합니다. 현재: {self.dropout}")
        
        if self.embedding_dim <= 0 or self.num_classes <= 0:
            raise ValueError(f"embedding_dim과 num_classes는 양수여야 합니다.")
        
        if self.mil_hidden_dim <= 0:
            raise ValueError(f"mil_hidden_dim은 양수여야 합니다.")
    
    def to_dict(self) -> Dict[str, Any]:
        """설정을 딕셔너리로 변환"""
        return {
            'embedding_dim': self.embedding_dim,
            'num_classes': self.num_classes,
            'mil_hidden_dim': self.mil_hidden_dim,
            'use_gated_attention': self.use_gated_attention,
            'dropout': self.dropout,
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'MILModelConfig':
        """딕셔너리에서 설정 생성"""
        return cls(**config_dict)
    
    def summary(self) -> str:
        """설정 요약 출력"""
        lines = [
            "=" * 60,
            "📋 MIL Model Configuration",
            "=" * 60,
            f"  입력 임베딩 차원: {self.embedding_dim}",
            f"  출력 클래스 수: {self.num_classes:,}",
            "",
            "  🔹 MIL 구조 설정:",
            f"     Hidden 차원: {self.mil_hidden_dim}",
            f"     Gated Attention: {'✅' if self.use_gated_attention else '❌'}",
            "",
            "  🔹 Regularization:",
            f"     Dropout: {self.dropout}",
            "=" * 60,
        ]
        return "\n".join(lines)


class AttentionPooling(nn.Module):
    """
    Standard Attention Pooling for MIL
    
    Attention mechanism:
        e = W^T * tanh(V * h)
        alpha = softmax(e)
        z = sum(alpha_i * h_i)
    
    Args:
        input_dim: 입력 차원 (embedding_dim)
        hidden_dim: Attention hidden 차원
    """
    def __init__(self, input_dim: int, hidden_dim: int):
        super(AttentionPooling, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # Attention layers
        self.attention_V = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh()
        )
        self.attention_U = nn.Linear(hidden_dim, 1)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> tuple:
        """
        Args:
            x: (batch, num_instances, input_dim) - chunk embeddings
            mask: (batch, num_instances) - 1=valid, 0=padding (optional)
        
        Returns:
            aggregated: (batch, input_dim) - aggregated representation
            attention_weights: (batch, num_instances) - attention weights
        """
        # Compute attention scores
        # V: (batch, num_instances, hidden_dim)
        V = self.attention_V(x)
        
        # U: (batch, num_instances, 1)
        U = self.attention_U(V)
        
        # Apply mask if provided (mask out padding)
        if mask is not None:
            # mask: (batch, num_instances) -> (batch, num_instances, 1)
            mask = mask.unsqueeze(-1)
            U = U.masked_fill(mask == 0, float('-inf'))
        
        # Attention weights: (batch, num_instances, 1)
        attention_weights = torch.softmax(U, dim=1)
        
        # Weighted sum: (batch, num_instances, input_dim) * (batch, num_instances, 1)
        aggregated = torch.sum(attention_weights * x, dim=1)  # (batch, input_dim)
        
        # Squeeze attention weights: (batch, num_instances)
        attention_weights = attention_weights.squeeze(-1)
        
        return aggregated, attention_weights


class GatedAttentionPooling(nn.Module):
    """
    Gated Attention Pooling for MIL (성능이 더 좋음)
    
    Gated Attention mechanism:
        e = (W^T * tanh(V * h)) ⊙ sigmoid(U * h)
        alpha = softmax(e)
        z = sum(alpha_i * h_i)
    
    where ⊙ is element-wise multiplication
    
    Args:
        input_dim: 입력 차원 (embedding_dim)
        hidden_dim: Attention hidden 차원
    """
    def __init__(self, input_dim: int, hidden_dim: int):
        super(GatedAttentionPooling, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # Attention layers
        self.attention_V = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh()
        )
        self.attention_U = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Sigmoid()
        )
        self.attention_W = nn.Linear(hidden_dim, 1)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> tuple:
        """
        Args:
            x: (batch, num_instances, input_dim) - chunk embeddings
            mask: (batch, num_instances) - 1=valid, 0=padding (optional)
        
        Returns:
            aggregated: (batch, input_dim) - aggregated representation
            attention_weights: (batch, num_instances) - attention weights
        """
        # Compute attention components
        # V: (batch, num_instances, hidden_dim)
        V = self.attention_V(x)
        
        # U: (batch, num_instances, hidden_dim)
        U = self.attention_U(x)
        
        # Gated attention: element-wise multiplication
        # (batch, num_instances, hidden_dim)
        gated = V * U
        
        # Attention scores: (batch, num_instances, 1)
        attention_scores = self.attention_W(gated)
        
        # Apply mask if provided (mask out padding)
        if mask is not None:
            # mask: (batch, num_instances) -> (batch, num_instances, 1)
            mask = mask.unsqueeze(-1)
            attention_scores = attention_scores.masked_fill(mask == 0, float('-inf'))
        
        # Attention weights: (batch, num_instances, 1)
        attention_weights = torch.softmax(attention_scores, dim=1)
        
        # Weighted sum: (batch, num_instances, input_dim) * (batch, num_instances, 1)
        aggregated = torch.sum(attention_weights * x, dim=1)  # (batch, input_dim)
        
        # Squeeze attention weights: (batch, num_instances)
        attention_weights = attention_weights.squeeze(-1)
        
        return aggregated, attention_weights


class AttentionMILClassifier(nn.Module):
    """
    Attention-based Multiple Instance Learning (MIL) Classifier
    
    Architecture:
    1. Chunk Embedding Processing (optional projection)
    2. Attention Aggregation (Gated or Standard)
    3. Classification Head
    
    Usage:
        # 방법 1: 직접 파라미터 전달
        model = AttentionMILClassifier(
            embedding_dim=2560,
            num_classes=40122,
            mil_hidden_dim=512,
            use_gated_attention=True,
            dropout=0.3
        )
        
        # 방법 2: MILModelConfig 사용
        config = MILModelConfig(...)
        model = AttentionMILClassifier.from_config(config)
    """
    
    def __init__(
        self,
        embedding_dim: int = 2560,
        num_classes: int = 40122,
        mil_hidden_dim: int = 512,
        use_gated_attention: bool = True,
        dropout: float = 0.3
    ):
        """
        Args:
            embedding_dim: ESM2-3B 임베딩 차원 (2560)
            num_classes: GO Term 수
            mil_hidden_dim: Attention 망과 분류기의 내부 차원
            use_gated_attention: True면 Gated Attention 사용 (성능이 더 좋음)
            dropout: Dropout 비율
        """
        super(AttentionMILClassifier, self).__init__()
        
        # 설정 저장
        self.embedding_dim = embedding_dim
        self.num_classes = num_classes
        self.mil_hidden_dim = mil_hidden_dim
        self.use_gated_attention = use_gated_attention
        self.dropout = dropout
        
        # =====================
        # 1. Optional Embedding Projection
        # =====================
        # 임베딩 차원을 hidden_dim으로 프로젝션 (선택적, 성능 향상 가능)
        self.embedding_proj = nn.Sequential(
            nn.Linear(embedding_dim, mil_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # =====================
        # 2. Attention Aggregation
        # =====================
        if use_gated_attention:
            self.attention = GatedAttentionPooling(
                input_dim=mil_hidden_dim,
                hidden_dim=mil_hidden_dim
            )
        else:
            self.attention = AttentionPooling(
                input_dim=mil_hidden_dim,
                hidden_dim=mil_hidden_dim
            )
        
        # =====================
        # 3. Classification Head
        # =====================
        self.classifier = nn.Sequential(
            nn.Linear(mil_hidden_dim, mil_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(mil_hidden_dim, num_classes)
        )
    
    @classmethod
    def from_config(cls, config: MILModelConfig) -> 'AttentionMILClassifier':
        """
        MILModelConfig에서 모델 생성
        
        Args:
            config: MILModelConfig 인스턴스
        
        Returns:
            AttentionMILClassifier 인스턴스
        """
        return cls(
            embedding_dim=config.embedding_dim,
            num_classes=config.num_classes,
            mil_hidden_dim=config.mil_hidden_dim,
            use_gated_attention=config.use_gated_attention,
            dropout=config.dropout
        )
    
    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: 입력 임베딩 (batch, max_chunks, embedding_dim)
                - 가장 긴 chunk 길이에 맞추고 빈 곳은 zero padding
            attention_mask: 어텐션 마스크 (batch, max_chunks) - 1=valid, 0=padding
        
        Returns:
            logits: (batch, num_classes)
        """
        # Project embeddings: (batch, max_chunks, embedding_dim) -> (batch, max_chunks, mil_hidden_dim)
        x = self.embedding_proj(x)
        
        # Attention aggregation: (batch, max_chunks, mil_hidden_dim) -> (batch, mil_hidden_dim)
        aggregated, attention_weights = self.attention(x, mask=attention_mask)
        
        # Classification: (batch, mil_hidden_dim) -> (batch, num_classes)
        logits = self.classifier(aggregated)
        
        return logits
    
    def get_config(self) -> Dict[str, Any]:
        """현재 모델 설정 반환"""
        return {
            'embedding_dim': self.embedding_dim,
            'num_classes': self.num_classes,
            'mil_hidden_dim': self.mil_hidden_dim,
            'use_gated_attention': self.use_gated_attention,
            'dropout': self.dropout,
        }
    
    def count_parameters(self) -> Dict[str, int]:
        """파라미터 수 계산"""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {
            'total': total,
            'trainable': trainable,
            'non_trainable': total - trainable
        }
    
    def summary(self) -> str:
        """모델 요약 출력"""
        params = self.count_parameters()
        param_memory = params['total'] * 4 / (1024**2)  # float32 = 4 bytes
        
        lines = [
            "=" * 60,
            "📐 Model Summary: AttentionMILClassifier",
            "=" * 60,
            "",
            "🔹 Architecture:",
            f"   Attention Type: {'Gated Attention' if self.use_gated_attention else 'Standard Attention'}",
            f"   Hidden Dimension: {self.mil_hidden_dim}",
            "",
            "🔹 Input/Output:",
            f"   Input: (batch, max_chunks, {self.embedding_dim})",
            f"   Output: (batch, {self.num_classes})",
            "",
            "📊 Parameters:",
            f"   Total: {params['total']:,}",
            f"   Trainable: {params['trainable']:,}",
            f"   Memory (est.): {param_memory:.2f} MB",
            "=" * 60,
        ]
        return "\n".join(lines)


# =========================================================================
# 편의 함수
# =========================================================================

def create_mil_model(
    embedding_dim: int = 2560,
    num_classes: int = 40122,
    mil_hidden_dim: int = 512,
    use_gated_attention: bool = True,
    dropout: float = 0.3,
    device: str = None
) -> AttentionMILClassifier:
    """
    MIL 모델 생성 편의 함수
    
    Args:
        embedding_dim: 임베딩 차원
        num_classes: 클래스 수
        mil_hidden_dim: MIL hidden 차원
        use_gated_attention: Gated Attention 사용 여부
        dropout: Dropout 비율
        device: 디바이스 ('cuda', 'cpu', None=자동)
    
    Returns:
        모델 인스턴스 (지정된 디바이스로 이동됨)
    """
    model = AttentionMILClassifier(
        embedding_dim=embedding_dim,
        num_classes=num_classes,
        mil_hidden_dim=mil_hidden_dim,
        use_gated_attention=use_gated_attention,
        dropout=dropout
    )
    
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    return model.to(device)


def test_mil_model(
    model: AttentionMILClassifier,
    max_chunks: int = 10,
    batch_size: int = 2
):
    """
    MIL 모델 테스트 (더미 입력으로 forward pass 확인)
    
    Args:
        model: 테스트할 모델
        max_chunks: 최대 chunk 수
        batch_size: 배치 크기
    """
    device = next(model.parameters()).device
    
    # 더미 입력: (batch, max_chunks, embedding_dim)
    dummy_input = torch.randn(batch_size, max_chunks, model.embedding_dim).to(device)
    
    # 더미 마스크: (batch, max_chunks) - 일부는 valid, 일부는 padding
    dummy_mask = torch.ones(batch_size, max_chunks).to(device)
    dummy_mask[:, max_chunks//2:] = 0  # 절반은 padding으로 설정
    
    with torch.no_grad():
        dummy_output = model(dummy_input, dummy_mask)
    
    print("🔍 MIL 모델 테스트:")
    print(f"   입력 shape: {dummy_input.shape}")
    print(f"   마스크 shape: {dummy_mask.shape}")
    print(f"   출력 shape: {dummy_output.shape}")
    print(f"   ✅ Forward pass 성공!")


# =========================================================================
# 손실 함수 (기존 model.py에서 가져옴)
# =========================================================================

# 기존 model.py의 손실함수들을 import하여 사용
# 순환 import 방지를 위해 여기서 직접 import하지 않고,
# 사용 시에 from model import AsymmetricLoss, ICWeightedBCELoss 등으로 import

# 손실함수는 기존 model.py에 정의되어 있으므로 그대로 사용 가능:
# - AsymmetricLoss
# - ICWeightedBCELoss
# - ICWeightedBCEWithLogitsLoss
# - load_ia_weights
# - create_ic_weighted_criterion

# =========================================================================
# 메인 (테스트용)
# =========================================================================
if __name__ == "__main__":
    print("=" * 70)
    print("🧪 Attention MIL Model Module Test")
    print("=" * 70)
    
    # 설정 테스트
    config = MILModelConfig(
        embedding_dim=2560,
        num_classes=1000,  # 테스트용 작은 값
        mil_hidden_dim=512,
        use_gated_attention=True,
        dropout=0.3
    )
    print(config.summary())
    
    # 모델 생성 테스트
    model = AttentionMILClassifier.from_config(config)
    print(model.summary())
    
    # Forward pass 테스트
    test_mil_model(model, max_chunks=10, batch_size=2)
    
    print("\n✅ 모든 테스트 성공!")

