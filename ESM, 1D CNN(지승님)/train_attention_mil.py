"""
CAFA 6 - Attention-based MIL Model Training Script for Azure ML
==================================================================
Azure ML Studio에서 컴퓨팅 클러스터를 사용한 MIL 모델 학습 스크립트

주요 기능:
- Azure ML 데이터셋 자동 마운트
- MLflow 자동 로깅 (metrics, parameters, model)
- Checkpoint 자동 저장 및 복구
- Early Stopping & Learning Rate Scheduling
- Chunk 기반 데이터셋 지원
"""

import os
import sys
import argparse
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from scipy.sparse import load_npz
from pathlib import Path
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm
import time
import json
import h5py

# Azure ML & MLflow
import mlflow
import mlflow.pytorch

# 로컬 모듈 임포트
from attention_mil_model import (
    AttentionMILClassifier,
    MILModelConfig,
)
from model import (
    TrainingConfig,
    train_epoch,
    validate_epoch,
    AsymmetricLoss,
    load_ia_weights,
    create_optimizer,
    create_scaler,
    predict,
)


# =========================================================================
# 커스텀 스케줄러: CosineAnnealingWarmRestarts with LR Decay
# =========================================================================

class CosineAnnealingWarmRestartsWithDecay(torch.optim.lr_scheduler._LRScheduler):
    """
    CosineAnnealingWarmRestarts의 개선 버전
    
    각 재시작 시 최대 학습률이 이전의 restart_decay_rate 비율로 감소
    예: restart_decay_rate=0.8이면, 재시작 시 이전 최대 LR의 80%로 시작
    
    Args:
        optimizer: 최적화 도구
        T_0: 첫 번째 재시작까지의 에폭 수
        T_mult: 재시작 주기 배수 (기본값: 2)
        eta_min: 최소 학습률 (기본값: 0)
        restart_decay_rate: 재시작 시 최대 LR 감소 비율 (기본값: 1.0, 즉 감소 없음)
        last_epoch: 마지막 에폭 번호 (기본값: -1)
    
    Example:
        >>> scheduler = CosineAnnealingWarmRestartsWithDecay(
        ...     optimizer, T_0=10, T_mult=2, 
        ...     eta_min=1e-7, restart_decay_rate=0.8
        ... )
        >>> # 첫 재시작: lr_max = base_lr
        >>> # 두 번째 재시작: lr_max = base_lr * 0.8
        >>> # 세 번째 재시작: lr_max = base_lr * 0.8^2 = base_lr * 0.64
    """
    
    def __init__(
        self, 
        optimizer, 
        T_0, 
        T_mult=2, 
        eta_min=0, 
        restart_decay_rate=1.0,
        last_epoch=-1
    ):
        if T_0 <= 0 or not isinstance(T_0, int):
            raise ValueError(f"Expected positive integer T_0, but got {T_0}")
        if T_mult < 1 or not isinstance(T_mult, int):
            raise ValueError(f"Expected integer T_mult >= 1, but got {T_mult}")
        if not 0.0 < restart_decay_rate <= 1.0:
            raise ValueError(f"restart_decay_rate should be in (0.0, 1.0], but got {restart_decay_rate}")
        
        self.T_0 = T_0
        self.T_mult = T_mult
        self.eta_min = eta_min
        self.restart_decay_rate = restart_decay_rate
        
        # 현재 주기 관련 변수
        self.T_cur = last_epoch
        self.T_i = T_0  # 현재 주기의 길이
        self.restart_count = 0  # 재시작 횟수
        
        # 기본 학습률 저장 (초기 최대 학습률)
        self.base_lrs = [group['lr'] for group in optimizer.param_groups]
        self.current_max_lrs = self.base_lrs.copy()  # 현재 주기의 최대 학습률
        
        # PyTorch의 _LRScheduler는 optimizer와 last_epoch만 받음
        super(CosineAnnealingWarmRestartsWithDecay, self).__init__(
            optimizer, last_epoch
        )
    
    def get_lr(self):
        """현재 에폭의 학습률 계산"""
        if self.T_cur == -1:
            return self.base_lrs
        
        # Cosine annealing 공식
        return [
            self.eta_min + (max_lr - self.eta_min) * 
            (1 + np.cos(np.pi * self.T_cur / self.T_i)) / 2
            for max_lr in self.current_max_lrs
        ]
    
    def step(self, epoch=None):
        """스케줄러 업데이트"""
        if epoch is None:
            epoch = self.last_epoch + 1
            self.T_cur = self.T_cur + 1
        else:
            self.T_cur = epoch
        
        self.last_epoch = epoch
        
        # 재시작 체크
        if self.T_cur >= self.T_i:
            self.restart_count += 1
            self.T_cur = 0
            self.T_i = self.T_i * self.T_mult
            
            # 재시작 시 최대 학습률 감소
            self.current_max_lrs = [
                max_lr * self.restart_decay_rate 
                for max_lr in self.current_max_lrs
            ]
        
        # 학습률 업데이트
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr
        
        self._last_lr = [group['lr'] for group in self.optimizer.param_groups]


def create_scheduler_with_decay(optimizer, config, scheduler_type='cosine_warm_restarts', restart_decay_rate=1.0):
    """
    학습률 스케줄러 생성 (재시작 감소율 지원)
    
    Args:
        optimizer: 최적화 도구
        config: TrainingConfig 객체
        scheduler_type: 스케줄러 타입
        restart_decay_rate: CosineWarmRestarts의 재시작 시 LR 감소 비율
    
    Returns:
        scheduler: 학습률 스케줄러
    """
    if scheduler_type == 'reduce_on_plateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=config.scheduler_factor,
            patience=config.scheduler_patience,
            min_lr=config.scheduler_min_lr,
            verbose=True
        )
    elif scheduler_type == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config.num_epochs,
            eta_min=config.scheduler_min_lr
        )
    elif scheduler_type == 'cosine_warm_restarts':
        # 재시작 감소율 적용된 커스텀 스케줄러
        scheduler = CosineAnnealingWarmRestartsWithDecay(
            optimizer,
            T_0=config.cosine_t_0,
            T_mult=config.cosine_t_mult,
            eta_min=config.scheduler_min_lr,
            restart_decay_rate=restart_decay_rate
        )
    elif scheduler_type == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=10,
            gamma=config.scheduler_factor
        )
    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")
    
    return scheduler


class ChunkEmbeddingDataset(Dataset):
    """
    Chunk 기반 임베딩 데이터셋 (MIL 모델용)
    
    Singleton Pattern per Worker로 각 워커별 파일 핸들 관리
    동적 패딩을 위해 collate_fn에서 패딩 처리
    
    HDF5 파일 구조:
    ├── embeddings (total_chunks, embedding_dim) - 모든 chunk 임베딩
    ├── chunk_offsets (num_sequences,) - 각 시퀀스의 chunk 시작 인덱스
    ├── chunk_counts (num_sequences,) - 각 시퀀스의 chunk 개수
    ├── ids (num_sequences,) - 단백질 ID
    └── seq_lengths (num_sequences,) - 원본 시퀀스 길이
    """
    
    # 워커별 파일 핸들 (Singleton Pattern)
    _worker_file_handles = {}
    
    def __init__(
        self,
        h5_path: Path,
        indices: np.ndarray = None,
        labels: np.ndarray = None
    ):
        """
        Args:
            h5_path: HDF5 파일 경로
            indices: 사용할 시퀀스 인덱스 배열 (None이면 전체 사용)
            labels: 라벨 매트릭스 (num_samples, num_classes)
        """
        self.h5_path = Path(h5_path)
        
        # 메타데이터 로드 (한 번만)
        with h5py.File(self.h5_path, 'r') as f:
            self.num_sequences = f.attrs.get('num_sequences', len(f['ids']))
            self.embedding_dim = f.attrs.get('embedding_dim', f['embeddings'].shape[1])
            
            # 전체 메타데이터 로드
            self.chunk_offsets = f['chunk_offsets'][:]
            self.chunk_counts = f['chunk_counts'][:]
            self.ids = f['ids'][:]
        
        # 인덱스 필터링
        if indices is not None:
            self.indices = indices
            self.chunk_offsets = self.chunk_offsets[indices]
            self.chunk_counts = self.chunk_counts[indices]
            self.ids = self.ids[indices]
        else:
            self.indices = np.arange(self.num_sequences)
        
        self.labels = labels
        
        print(f"✅ ChunkEmbeddingDataset 초기화:")
        print(f"   시퀀스 수: {len(self.indices):,}")
        print(f"   임베딩 차원: {self.embedding_dim}")
        print(f"   최대 chunk 수: {int(self.chunk_counts.max())}")
        print(f"   평균 chunk/시퀀스: {self.chunk_counts.mean():.2f}")
    
    def _get_file_handle(self):
        """
        워커별 파일 핸들 가져오기 (Singleton Pattern per Worker)
        각 워커는 자신만의 파일 핸들을 가짐 (Thread-safe)
        """
        worker_info = torch.utils.data.get_worker_info()
        worker_id = worker_info.id if worker_info is not None else 0
        
        # 워커별 파일 핸들이 없으면 생성 (Lazy Loading)
        if worker_id not in self._worker_file_handles:
            self._worker_file_handles[worker_id] = h5py.File(self.h5_path, 'r')
        
        return self._worker_file_handles[worker_id]
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        """
        Returns:
            embedding: (num_chunks, embedding_dim) - 원본 chunk 임베딩 (패딩 없음)
            num_chunks: int - 실제 chunk 개수
            label: (num_classes,) - 라벨 벡터
        """
        # 워커별 파일 핸들 가져오기
        f = self._get_file_handle()
        
        # Chunk 임베딩 로드
        offset = self.chunk_offsets[idx]
        count = self.chunk_counts[idx]
        chunks = f['embeddings'][offset:offset + count][:]  # (count, embedding_dim)
        
        # 라벨
        label = self.labels[idx] if self.labels is not None else np.zeros(1)
        
        return {
            'embedding': torch.from_numpy(chunks.astype(np.float32)),
            'num_chunks': int(count),
            'label': torch.from_numpy(label.astype(np.float32))
        }


def collate_fn_mil(batch):
    """
    MIL 모델용 동적 패딩 collate 함수
    
    배치 내에서 가장 긴 시퀀스 길이에 맞춰 동적으로 패딩
    메모리 효율적이고 배치별로 최적화됨
    """
    # 배치 내 최대 chunk 수 계산
    max_chunks = max(item['num_chunks'] for item in batch)
    batch_size = len(batch)
    embedding_dim = batch[0]['embedding'].shape[1]
    
    # 패딩된 텐서 초기화
    embeddings = torch.zeros(batch_size, max_chunks, embedding_dim, dtype=torch.float32)
    attention_masks = torch.zeros(batch_size, max_chunks, dtype=torch.float32)
    labels = torch.stack([item['label'] for item in batch])
    
    # 각 샘플을 패딩하여 채움
    for i, item in enumerate(batch):
        num_chunks = item['num_chunks']
        embeddings[i, :num_chunks, :] = item['embedding']
        attention_masks[i, :num_chunks] = 1.0
    
    return {
        'embedding': embeddings,
        'attention_mask': attention_masks,
        'label': labels
    }


def worker_init_fn(worker_id):
    """
    DataLoader 워커 초기화 함수
    
    각 워커가 자신의 random seed를 가지도록 설정
    Singleton Pattern으로 파일 핸들 관리
    """
    worker_info = torch.utils.data.get_worker_info()
    if worker_info is not None:
        # 각 워커별로 다른 random seed 설정
        np.random.seed(worker_info.seed % (2**32))
        # 파일 핸들은 Dataset의 _get_file_handle()에서 Lazy Loading됨


def calculate_fmax_gpu(
    preds: torch.Tensor,
    labels: torch.Tensor,
    ia_weights: torch.Tensor,
    thresholds: torch.Tensor = None,
    device: torch.device = None
):
    """
    GPU 가속 Weighted F-max 계산 (CAFA6 공식 평가 지표)
    
    Args:
        preds: (N_proteins, N_terms) 예측 확률 (0~1) - torch.Tensor
        labels: (N_proteins, N_terms) 실제 라벨 (0 또는 1) - torch.Tensor
        ia_weights: (N_terms,) IC 가중치 - torch.Tensor
        thresholds: 스캔할 임계값 텐서 (기본: 0~1, 51개)
        device: GPU 디바이스
    
    Returns:
        best_fmax: 최고 F-max 점수 (float)
        best_threshold: 최적 임계값 (float)
        all_f1_scores: 모든 임계값에서의 F1 점수 리스트
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 모든 텐서를 GPU로 이동
    preds = preds.to(device)
    labels = labels.to(device)
    ia_weights = ia_weights.to(device)
    
    if thresholds is None:
        thresholds = torch.linspace(0, 1, 51, device=device)
    else:
        thresholds = thresholds.to(device)
    
    # Ground Truth의 총 IC (Recall 분모)
    # labels: (N_proteins, N_terms), ia_weights: (N_terms,)
    # total_true_ic: (N_proteins,)
    total_true_ic = torch.matmul(labels, ia_weights)
    total_true_ic = torch.clamp(total_true_ic, min=1e-10)  # Division by zero 방지
    
    best_fmax = 0.0
    best_threshold = 0.0
    all_f1_scores = []
    
    # 모든 임계값에 대해 배치 계산
    for t in thresholds:
        # 1. 임계값 적용
        pred_binary = (preds >= t).float()  # (N_proteins, N_terms)
        
        # 2. True Positive (교집합)
        tp_mask = pred_binary * labels  # (N_proteins, N_terms)
        weighted_tp = torch.matmul(tp_mask, ia_weights)  # (N_proteins,)
        
        # 3. Prediction의 총 IC (Precision 분모)
        weighted_pred = torch.matmul(pred_binary, ia_weights)  # (N_proteins,)
        
        # 4. Weighted Precision & Recall
        # Precision: weighted_tp / weighted_pred
        precision = torch.where(
            weighted_pred > 0,
            weighted_tp / weighted_pred,
            torch.zeros_like(weighted_tp)
        )
        
        # Recall: weighted_tp / total_true_ic
        recall = weighted_tp / total_true_ic
        
        # 5. 평균
        avg_prec = precision.mean()
        avg_rec = recall.mean()
        
        # 6. F1 계산
        if (avg_prec + avg_rec) > 0:
            f1 = 2 * avg_prec * avg_rec / (avg_prec + avg_rec)
        else:
            f1 = torch.tensor(0.0, device=device)
        
        f1_value = f1.item()
        all_f1_scores.append(f1_value)
        
        # 7. 최고 F-max 업데이트
        if f1_value > best_fmax:
            best_fmax = f1_value
            best_threshold = t.item()
    
    return best_fmax, best_threshold, all_f1_scores


def parse_args():
    """커맨드라인 인자 파싱"""
    parser = argparse.ArgumentParser(description='CAFA 6 - Attention MIL Training on Azure ML')
    
    # ==================== Azure ML 데이터셋 ====================
    parser.add_argument('--data_path', type=str, required=True,
                        help='Azure ML 데이터셋 마운트 경로')
    
    parser.add_argument('--dataset_name', type=str,
                        default='CAFA6_82404sq_Asymmetric Loss',
                        help='데이터셋 폴더 이름')
    
    # ==================== 모델 아키텍처 ====================
    parser.add_argument('--embedding_dim', type=int, default=2560,
                        help='ESM2-3B 임베딩 차원')
    parser.add_argument('--num_classes', type=int, default=40122,
                        help='GO Term 클래스 수')
    parser.add_argument('--mil_hidden_dim', type=int, default=512,
                        help='MIL hidden 차원')
    parser.add_argument('--use_gated_attention', action='store_true', default=True,
                        help='Gated Attention 사용')
    parser.add_argument('--dropout', type=float, default=0.3,
                        help='Dropout 비율')
    parser.add_argument('--max_chunks', type=int, default=None,
                        help='최대 chunk 수 (None이면 자동 계산)')
    
    # ==================== 학습 설정 ====================
    parser.add_argument('--batch_size', type=int, default=32,
                        help='배치 크기')
    parser.add_argument('--num_epochs', type=int, default=50,
                        help='학습 에폭 수')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                        help='학습률')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                        help='Weight decay (L2 정규화)')
    parser.add_argument('--grad_clip_norm', type=float, default=1.0,
                        help='Gradient clipping norm')
    
    # ==================== 손실 함수 ====================
    parser.add_argument('--use_ic_weighted_loss', action='store_true', default=True,
                        help='IC 가중치 기반 AsymmetricLoss 사용')
    parser.add_argument('--gamma_pos', type=float, default=1.0,
                        help='AsymmetricLoss gamma_pos')
    parser.add_argument('--gamma_neg', type=float, default=4.0,
                        help='AsymmetricLoss gamma_neg')
    parser.add_argument('--asl_clip', type=float, default=0.05,
                        help='AsymmetricLoss clip value')
    
    # ==================== 스케줄러 & Early Stopping ====================
    parser.add_argument('--scheduler_type', type=str, default='reduce_on_plateau',
                        choices=['reduce_on_plateau', 'cosine', 'step', 'cosine_warm_restarts'],
                        help='스케줄러 타입')
    parser.add_argument('--scheduler_factor', type=float, default=0.5,
                        help='LR 감소 비율')
    parser.add_argument('--scheduler_patience', type=int, default=3,
                        help='LR 스케줄러 patience')
    parser.add_argument('--scheduler_min_lr', type=float, default=1e-7,
                        help='최소 학습률')
    parser.add_argument('--cosine_t_0', type=int, default=10,
                        help='CosineAnnealingWarmRestarts의 첫 번째 restart까지 에폭 수')
    parser.add_argument('--cosine_t_mult', type=int, default=2,
                        help='CosineAnnealingWarmRestarts의 restart 주기 배수')
    parser.add_argument('--cosine_restart_decay', type=float, default=0.8,
                        help='CosineWarmRestarts 재시작 시 LR 감소 비율 (0.8 = 이전 최대 LR의 80%)')
    parser.add_argument('--early_stop_patience', type=int, default=7,
                        help='Early stopping patience')
    parser.add_argument('--early_stop_min_delta', type=float, default=1e-4,
                        help='Early stopping 최소 개선 기준')
    
    # ==================== DataLoader ====================
    parser.add_argument('--num_workers', type=int, default=8,
                        help='DataLoader worker 수')
    parser.add_argument('--val_split', type=float, default=0.1,
                        help='Validation split 비율')
    parser.add_argument('--random_seed', type=int, default=42,
                        help='Random seed')
    
    # ==================== 최적화 ====================
    parser.add_argument('--use_amp', action='store_true', default=True,
                        help='AMP (Automatic Mixed Precision) 사용')
    
    # ==================== 출력 ====================
    parser.add_argument('--output_dir', type=str, default='./outputs',
                        help='모델 저장 디렉토리')
    parser.add_argument('--log_interval', type=int, default=10,
                        help='로그 출력 주기 (배치 단위)')
    
    args = parser.parse_args()
    return args


def setup_paths(args):
    """Azure ML 데이터셋 경로 설정"""
    data_path = Path(args.data_path)
    
    print("=" * 70)
    print("📂 데이터셋 경로 검증")
    print("=" * 70)
    print(f"🔍 data_path: {data_path}")
    print(f"🔍 dataset_name: {args.dataset_name}")
    
    # 데이터셋 루트 경로 결정
    expected_folders = ['embedded_fasta_esm2_3b', 'go_basic', 'cafa-6-protein-function-prediction']
    has_expected_structure = any((data_path / folder).exists() for folder in expected_folders)
    
    if has_expected_structure:
        base_dir = data_path
        print(f"\n✅ 감지: data_path가 이미 데이터셋 루트")
    else:
        potential_base = data_path / args.dataset_name
        if potential_base.exists() and any((potential_base / folder).exists() for folder in expected_folders):
            base_dir = potential_base
            print(f"\n✅ 감지: {args.dataset_name} 하위 폴더가 데이터셋 루트")
        else:
            base_dir = data_path
            print(f"\n⚠️ 예상 폴더 구조를 찾지 못함, data_path 사용")
    
    print(f"🔍 최종 base_dir: {base_dir}")
    
    # Chunk 임베딩 파일 경로 (ESM2-3B)
    paths = {
        'base_dir': base_dir,
        'train_chunk_h5': base_dir / 'embedded_fasta_esm2_3b' / 'train' / 'esm2_3b_chunk_embeddings_82404sq.h5',
        'go_dir': base_dir / 'go_basic',
        'go_mappings': base_dir / 'go_basic' / 'go_term_mappings.pkl',
        'ancestor_matrix': base_dir / 'go_basic' / 'go_ancestor_matrix.npz',
        'train_terms': base_dir / 'cafa-6-protein-function-prediction' / 'Train' / 'train_terms.tsv',
        'ia_tsv': base_dir / 'cafa-6-protein-function-prediction' / 'IA.tsv',
    }
    
    # 경로 검증
    print(f"\n📂 파일 경로 검증:")
    print("=" * 70)
    for name, path in paths.items():
        exists = path.exists() if isinstance(path, Path) else False
        status = "✅" if exists else "❌"
        print(f"{status} {name}: {path}")
        if not exists and name not in ['go_dir']:
            raise FileNotFoundError(f"필수 파일 없음: {path}")
    
    return paths


def load_go_mappings(paths):
    """GO Term 매핑 및 계층 구조 로드"""
    print("\n" + "=" * 70)
    print("📊 GO Term 매핑 로드")
    print("=" * 70)
    
    with open(paths['go_mappings'], 'rb') as f:
        go_mappings = pickle.load(f)
    
    term_to_idx = go_mappings['go_to_idx']
    idx_to_term = go_mappings['idx_to_go']
    num_go_terms = len(term_to_idx)
    
    print(f"✅ GO Term 총 개수: {num_go_terms:,}")
    
    ancestor_matrix = load_npz(paths['ancestor_matrix'])
    print(f"✅ GO ancestor matrix shape: {ancestor_matrix.shape}")
    
    return term_to_idx, idx_to_term, num_go_terms, ancestor_matrix


def extract_uniprot_accession(full_id):
    """
    UniProt 전체 ID에서 Accession만 추출
    
    Args:
        full_id: bytes 또는 str (예: b'sp|Q6UXS9|CASPC_HUMAN' 또는 'sp|Q6UXS9|CASPC_HUMAN')
    
    Returns:
        Accession만 추출된 ID (예: 'Q6UXS9')
    """
    # Bytes → String 디코딩
    if isinstance(full_id, (bytes, np.bytes_)):
        full_id = full_id.decode('utf-8')
    else:
        full_id = str(full_id)
    
    # UniProt 형식 파싱: 'sp|Q6UXS9|CASPC_HUMAN' → 'Q6UXS9'
    if '|' in full_id:
        parts = full_id.split('|')
        if len(parts) >= 2:
            return parts[1]  # sp|ACCESSION|NAME 형식에서 ACCESSION 추출
    
    # 이미 Accession만 있는 경우 그대로 반환
    return full_id


def load_training_labels(paths, train_ids, term_to_idx, num_go_terms, ancestor_matrix):
    """학습 라벨 로드 및 전파"""
    print("\n" + "=" * 70)
    print("🏷️ 학습 라벨 로드")
    print("=" * 70)
    
    # HDF5에서 읽은 ID를 Accession으로 변환
    # bytes 디코딩 + UniProt 형식 파싱 (sp|Q6UXS9|CASPC_HUMAN → Q6UXS9)
    print(f"📋 원본 ID 샘플: {repr(train_ids[0])}")
    
    train_ids_decoded = np.array([extract_uniprot_accession(id) for id in train_ids])
    
    print(f"✅ ID 변환 완료: {len(train_ids_decoded):,}개")
    print(f"   예시: {repr(train_ids[0])} → {repr(train_ids_decoded[0])}")
    print(f"   샘플 (처음 5개): {train_ids_decoded[:5].tolist()}")
    
    train_terms_df = pd.read_csv(paths['train_terms'], sep='\t')
    train_terms_df.columns = ['EntryID', 'GO_Term', 'Aspect']
    
    # EntryID를 문자열로 확실히 변환
    train_terms_df['EntryID'] = train_terms_df['EntryID'].astype(str)
    
    print(f"\n📊 Train terms 통계:")
    print(f"   총 라벨 개수: {len(train_terms_df):,}")
    print(f"   고유 단백질: {train_terms_df['EntryID'].nunique():,}")
    print(f"   TSV ID 샘플: {train_terms_df['EntryID'].unique()[:5].tolist()}")
    
    # 디코딩된 ID로 매핑 딕셔너리 생성
    protein_id_to_idx = {str(protein_id): idx for idx, protein_id in enumerate(train_ids_decoded)}
    
    # train_ids_decoded를 set으로 변환하여 빠른 조회
    train_ids_set = set(train_ids_decoded)
    
    print(f"\n🔄 ID 매칭 진행 중...")
    print(f"   HDF5 고유 ID 수: {len(train_ids_set):,}")
    print(f"   TSV 고유 단백질 수: {train_terms_df['EntryID'].nunique():,}")
    
    train_terms_df = train_terms_df[train_terms_df['EntryID'].isin(train_ids_set)].copy()
    
    print(f"✅ 매칭 성공:")
    print(f"   매칭된 라벨 수: {len(train_terms_df):,}")
    print(f"   매칭된 단백질 수: {train_terms_df['EntryID'].nunique():,}")
    
    num_proteins = len(train_ids_decoded)
    label_matrix = np.zeros((num_proteins, num_go_terms), dtype=np.float32)
    
    matched_count = 0
    for _, row in tqdm(train_terms_df.iterrows(), total=len(train_terms_df), desc="라벨 매트릭스 구축"):
        protein_id = str(row['EntryID'])  # 확실히 문자열로 변환
        go_term = row['GO_Term']
        
        if protein_id in protein_id_to_idx and go_term in term_to_idx:
            protein_idx = protein_id_to_idx[protein_id]
            go_idx = term_to_idx[go_term]
            label_matrix[protein_idx, go_idx] = 1.0
            matched_count += 1
    
    print(f"✅ 매칭 성공한 라벨: {matched_count:,}개")
    
    print(f"✅ Label matrix shape: {label_matrix.shape}")
    
    # 라벨 전파
    print("\n🔄 라벨 전파 (계층 구조 적용)...")
    label_matrix_propagated = propagate_labels_gpu(label_matrix, ancestor_matrix)
    print(f"✅ 전파 후 positive labels: {label_matrix_propagated.sum():.0f}")
    
    return label_matrix_propagated


def propagate_labels_gpu(label_matrix, ancestor_matrix, batch_size=4096):
    """GPU 가속 라벨 전파"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if hasattr(ancestor_matrix, 'toarray'):
        ancestor_T_dense = ancestor_matrix.T.toarray()
    else:
        ancestor_T_dense = ancestor_matrix.T
    
    ancestor_tensor_T = torch.tensor(ancestor_T_dense, dtype=torch.float32).to(device)
    
    num_proteins = label_matrix.shape[0]
    result_batches = []
    
    for start_idx in tqdm(range(0, num_proteins, batch_size), desc="라벨 전파"):
        end_idx = min(start_idx + batch_size, num_proteins)
        batch_input = label_matrix[start_idx:end_idx]
        batch_tensor = torch.tensor(batch_input, dtype=torch.float32).to(device)
        batch_result = torch.matmul(batch_tensor, ancestor_tensor_T)
        batch_result = (batch_result > 0).float()
        result_batches.append(batch_result.cpu().numpy())
        del batch_tensor, batch_result
    
    torch.cuda.empty_cache()
    return np.vstack(result_batches)


def create_dataloaders(args, train_chunk_h5_path, train_ids, label_matrix):
    """Train/Validation DataLoader 생성"""
    print("\n" + "=" * 70)
    print("🔄 DataLoader 생성")
    print("=" * 70)
    
    # Train/Val split
    num_samples = len(train_ids)
    train_idx, val_idx = train_test_split(
        np.arange(num_samples),
        test_size=args.val_split,
        random_state=args.random_seed
    )
    
    y_train = label_matrix[train_idx]
    y_val = label_matrix[val_idx]
    
    print(f"✅ Train: {len(train_idx):,} samples")
    print(f"✅ Validation: {len(val_idx):,} samples")
    
    # 데이터셋 생성 (동적 패딩 사용)
    train_dataset = ChunkEmbeddingDataset(
        h5_path=train_chunk_h5_path,
        indices=train_idx,
        labels=y_train
    )
    
    val_dataset = ChunkEmbeddingDataset(
        h5_path=train_chunk_h5_path,
        indices=val_idx,
        labels=y_val
    )
    
    # DataLoader 생성
    use_cuda = torch.cuda.is_available()
    pin_memory = use_cuda
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn_mil,
        worker_init_fn=worker_init_fn if args.num_workers > 0 else None,
        persistent_workers=True if args.num_workers > 0 else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn_mil,
        worker_init_fn=worker_init_fn if args.num_workers > 0 else None,
        persistent_workers=True if args.num_workers > 0 else False
    )
    
    print(f"✅ Batch size: {args.batch_size}")
    print(f"✅ Num workers: {args.num_workers}")
    
    return train_loader, val_loader


def setup_mlflow(args):
    """MLflow 실험 설정"""
    print("\n" + "=" * 70)
    print("📊 MLflow 설정")
    print("=" * 70)
    
    try:
        mlflow.set_experiment("CAFA6_Attention_MIL_Training")
        
        mlflow.log_params({
            'embedding_dim': args.embedding_dim,
            'mil_hidden_dim': args.mil_hidden_dim,
            'use_gated_attention': args.use_gated_attention,
            'dropout': args.dropout,
            'max_chunks': args.max_chunks,
            'batch_size': args.batch_size,
            'num_epochs': args.num_epochs,
            'learning_rate': args.learning_rate,
            'weight_decay': args.weight_decay,
            'grad_clip_norm': args.grad_clip_norm,
            'use_ic_weighted_loss': args.use_ic_weighted_loss,
            'gamma_pos': args.gamma_pos,
            'gamma_neg': args.gamma_neg,
            'asl_clip': args.asl_clip,
            'scheduler_type': args.scheduler_type,
            'cosine_t_0': args.cosine_t_0,
            'cosine_t_mult': args.cosine_t_mult,
            'cosine_restart_decay': args.cosine_restart_decay,
            'use_amp': args.use_amp,
            'num_workers': args.num_workers,
            'val_split': args.val_split,
            'random_seed': args.random_seed,
        })
        print("✅ MLflow 로깅 시작")
    except Exception as e:
        print(f"⚠️ MLflow 설정 실패 (계속 진행): {e}")


def main():
    """메인 학습 함수"""
    args = parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n{'='*70}")
    print(f"🚀 CAFA 6 - Attention MIL Training on Azure ML")
    print(f"{'='*70}")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    setup_mlflow(args)
    
    paths = setup_paths(args)
    
    term_to_idx, idx_to_term, num_go_terms, ancestor_matrix = load_go_mappings(paths)
    
    # Chunk 임베딩 메타데이터 로드
    print("\n" + "=" * 70)
    print("📥 Chunk 임베딩 메타데이터 로드")
    print("=" * 70)
    
    with h5py.File(paths['train_chunk_h5'], 'r') as f:
        train_ids = f['ids'][:]
        num_sequences = len(train_ids)
        max_chunks_in_data = int(f['chunk_counts'][:].max())
    
    print(f"✅ Train 시퀀스 수: {num_sequences:,}")
    print(f"✅ 데이터셋 최대 chunk 수: {max_chunks_in_data}")
    
    # IC 가중치 로드 (평가용)
    print("\n" + "=" * 70)
    print("📊 IC 가중치 로드 (평가용)")
    print("=" * 70)
    
    ia_weights_tensor = None
    if paths['ia_tsv'].exists():
        ia_weights_tensor = load_ia_weights(
            ia_path=str(paths['ia_tsv']),
            go_to_idx=term_to_idx,
            num_classes=num_go_terms,
            default_weight=0.0
        ).to(device)  # GPU로 이동
        print(f"✅ IC 가중치 로드 완료 (GPU에서 평가 수행)")
    else:
        print(f"⚠️ IA.tsv 없음 - 평가 지표 계산 생략")
    
    label_matrix = load_training_labels(
        paths, train_ids, term_to_idx, num_go_terms, ancestor_matrix
    )
    
    train_loader, val_loader = create_dataloaders(
        args, paths['train_chunk_h5'], train_ids, label_matrix
    )
    
    # 모델 생성
    print("\n" + "=" * 70)
    print("🏗️ 모델 생성")
    print("=" * 70)
    
    model_config = MILModelConfig(
        embedding_dim=args.embedding_dim,
        num_classes=num_go_terms,
        mil_hidden_dim=args.mil_hidden_dim,
        use_gated_attention=args.use_gated_attention,
        dropout=args.dropout
    )
    
    model = AttentionMILClassifier.from_config(model_config).to(device)
    print(model.summary())
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    try:
        mlflow.log_param('total_parameters', total_params)
        mlflow.log_param('trainable_parameters', trainable_params)
    except Exception as e:
        print(f"⚠️ MLflow 파라미터 로깅 실패: {e}")
    
    # 손실 함수
    print("\n" + "=" * 70)
    print("🔥 손실 함수 설정")
    print("=" * 70)
    
    if args.use_ic_weighted_loss and paths['ia_tsv'].exists():
        ic_weights = load_ia_weights(
            ia_path=str(paths['ia_tsv']),
            go_to_idx=term_to_idx,
            num_classes=num_go_terms,
            default_weight=0.0
        ).to(device)
        
        criterion = AsymmetricLoss(
            gamma_pos=args.gamma_pos,
            gamma_neg=args.gamma_neg,
            clip=args.asl_clip,
            ic_weights=ic_weights,
            normalize_ic=True,
            reduction='mean'
        )
        print(f"✅ AsymmetricLoss with IC weights")
    else:
        criterion = nn.BCEWithLogitsLoss()
        print(f"✅ BCEWithLogitsLoss")
    
    # 학습 설정
    training_config = TrainingConfig(
        use_amp=args.use_amp,
        grad_clip_norm=args.grad_clip_norm,
        non_blocking=True,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        batch_size=args.batch_size,
        scheduler_factor=args.scheduler_factor,
        scheduler_patience=args.scheduler_patience,
        scheduler_min_lr=args.scheduler_min_lr,
        cosine_t_0=args.cosine_t_0,
        cosine_t_mult=args.cosine_t_mult,
        early_stop_patience=args.early_stop_patience,
        early_stop_min_delta=args.early_stop_min_delta,
        use_pos_weight=False
    )
    
    optimizer = create_optimizer(model, config=training_config, optimizer_type='adamw')
    
    # 스케줄러 생성 (재시작 감소율 지원)
    scheduler = create_scheduler_with_decay(
        optimizer, 
        config=training_config, 
        scheduler_type=args.scheduler_type,
        restart_decay_rate=args.cosine_restart_decay
    )
    
    scaler = create_scaler(enabled=args.use_amp)
    
    print(f"✅ Optimizer: AdamW (lr={args.learning_rate}, wd={args.weight_decay})")
    print(f"✅ Scheduler: {args.scheduler_type}")
    if args.scheduler_type == 'cosine_warm_restarts':
        print(f"   T_0={args.cosine_t_0}, T_mult={args.cosine_t_mult}")
        print(f"   Restart LR decay: {args.cosine_restart_decay:.2%} (각 재시작 시 최대 LR이 {args.cosine_restart_decay:.0%}로 감소)")
    
    # 학습 루프
    print("\n" + "=" * 70)
    print("🚀 학습 시작")
    print("=" * 70)
    
    best_val_loss = float('inf')
    best_val_fmax = 0.0
    patience_counter = 0
    train_losses = []
    val_losses = []
    val_fmax_scores = []
    
    for epoch in range(args.num_epochs):
        epoch_start_time = time.time()
        print(f"\n{'='*70}")
        print(f"Epoch {epoch+1}/{args.num_epochs}")
        print(f"{'='*70}")
        
        train_loss = train_epoch(
            model, train_loader, optimizer, criterion, device,
            scaler=scaler, config=training_config
        )
        train_losses.append(train_loss)
        
        val_loss = validate_epoch(
            model, val_loader, criterion, device,
            config=training_config
        )
        val_losses.append(val_loss)
        
        # Validation F-max 계산 (매 에폭, GPU 가속)
        val_fmax = 0.0
        if ia_weights_tensor is not None and (epoch + 1) % 1 == 0:  # 매 에폭마다
            print(f"\n📈 Validation F-max 계산 중 (GPU)...")
            try:
                # Validation 예측 (GPU 텐서로 반환)
                val_preds = predict(
                    model, val_loader, device,
                    config=training_config,
                    return_tensor=True  # torch.Tensor로 반환
                )
                
                # Validation 라벨 (GPU 텐서로 변환)
                val_labels = torch.from_numpy(
                    label_matrix[val_loader.dataset.indices]
                ).to(device)
                
                # F-max 계산 (GPU)
                val_fmax, fmax_threshold, _ = calculate_fmax_gpu(
                    preds=val_preds,
                    labels=val_labels,
                    ia_weights=ia_weights_tensor,
                    thresholds=torch.linspace(0, 1, 51, device=device),  # GPU 텐서
                    device=device
                )
                val_fmax_scores.append(val_fmax)
                
                print(f"   Val F-max: {val_fmax:.4f} (threshold={fmax_threshold:.2f})")
            except Exception as e:
                print(f"   ⚠️ F-max 계산 실패: {e}")
                import traceback
                traceback.print_exc()
                val_fmax = 0.0
        
        if args.scheduler_type == 'reduce_on_plateau':
            scheduler.step(val_loss)
        else:
            scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        
        epoch_time = time.time() - epoch_start_time
        
        print(f"\n📊 Epoch {epoch+1} 결과:")
        print(f"   Train Loss: {train_loss:.6f}")
        print(f"   Val Loss: {val_loss:.6f}")
        if val_fmax > 0:
            print(f"   Val F-max: {val_fmax:.4f}")
        print(f"   LR: {current_lr:.2e}")
        print(f"   Time: {epoch_time:.1f}s")
        
        try:
            metrics = {
                'train_loss': train_loss,
                'val_loss': val_loss,
                'learning_rate': current_lr,
                'epoch_time': epoch_time
            }
            if val_fmax > 0:
                metrics['val_fmax'] = val_fmax
            mlflow.log_metrics(metrics, step=epoch)
        except Exception as e:
            print(f"   ⚠️ MLflow 메트릭 로깅 실패: {e}")
        
        # 모델 저장 (Val F-max 기준)
        # F-max가 계산된 경우 F-max 기준, 아니면 Loss 기준
        save_model = False
        improvement_msg = ""
        
        if ia_weights_tensor is not None and val_fmax > 0:
            # F-max 기준 (높을수록 좋음)
            if val_fmax > best_val_fmax * (1 + args.early_stop_min_delta):
                improvement = val_fmax - best_val_fmax
                best_val_fmax = val_fmax
                best_val_loss = val_loss
                save_model = True
                improvement_msg = f"Val F-max: {val_fmax:.4f} (improvement: +{improvement:.4f})"
                patience_counter = 0
            else:
                patience_counter += 1
                print(f"   ⏳ No improvement in F-max ({patience_counter}/{args.early_stop_patience})")
        else:
            # Loss 기준 (낮을수록 좋음) - F-max 계산 불가능할 때 fallback
            if val_loss < best_val_loss * (1 - args.early_stop_min_delta):
                improvement = best_val_loss - val_loss
                best_val_loss = val_loss
                if val_fmax > best_val_fmax:
                    best_val_fmax = val_fmax
                save_model = True
                improvement_msg = f"Val Loss: {val_loss:.6f} (improvement: -{improvement:.6f})"
                patience_counter = 0
            else:
                patience_counter += 1
                print(f"   ⏳ No improvement in Loss ({patience_counter}/{args.early_stop_patience})")
        
        if save_model:
            checkpoint_path = output_dir / 'best_model.pth'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'val_fmax': val_fmax,
                'best_val_fmax': best_val_fmax,
                'best_val_loss': best_val_loss,
                'model_config': model_config.to_dict(),
                'training_config': training_config.to_dict(),
                'args': vars(args),
            }, checkpoint_path)
            
            print(f"   ✅ Best model saved: {checkpoint_path}")
            print(f"   ✅ {improvement_msg}")
            if val_fmax > 0:
                print(f"   📊 Current - Loss: {val_loss:.6f}, F-max: {val_fmax:.4f}")
        
        if patience_counter >= args.early_stop_patience:
            print(f"\n⚠️ Early stopping at epoch {epoch+1}")
            try:
                mlflow.log_metric('early_stopped_epoch', epoch)
            except Exception as e:
                print(f"⚠️ MLflow 메트릭 로깅 실패: {e}")
            break
    
    print("\n" + "=" * 70)
    print("✅ 학습 완료!")
    print("=" * 70)
    print(f"Best validation loss: {best_val_loss:.6f}")
    if best_val_fmax > 0:
        print(f"Best validation F-max: {best_val_fmax:.4f}")
    print(f"Total epochs trained: {len(train_losses)}")
    print(f"모델 저장 경로: {output_dir / 'best_model.pth'}")
    
    try:
        final_metrics = {
            'best_val_loss': best_val_loss,
            'total_epochs': len(train_losses),
        }
        if best_val_fmax > 0:
            final_metrics['best_val_fmax'] = best_val_fmax
        mlflow.log_metrics(final_metrics)
    except Exception as e:
        print(f"⚠️ MLflow 최종 메트릭 로깅 실패: {e}")
    
    # 학습 곡선 저장
    try:
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(1, 2 if len(val_fmax_scores) > 0 else 1, figsize=(15 if len(val_fmax_scores) > 0 else 10, 6))
        
        if len(val_fmax_scores) > 0:
            # Loss plot
            axes[0].plot(train_losses, label='Train Loss')
            axes[0].plot(val_losses, label='Val Loss')
            axes[0].set_xlabel('Epoch')
            axes[0].set_ylabel('Loss')
            axes[0].set_title('Training History - Loss')
            axes[0].legend()
            axes[0].grid(True)
            
            # F-max plot
            axes[1].plot(val_fmax_scores, label='Val F-max', color='green')
            axes[1].set_xlabel('Epoch')
            axes[1].set_ylabel('F-max')
            axes[1].set_title('Training History - F-max')
            axes[1].legend()
            axes[1].grid(True)
        else:
            # Loss plot only
            ax = axes if isinstance(axes, plt.Axes) else axes
            ax.plot(train_losses, label='Train Loss')
            ax.plot(val_losses, label='Val Loss')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Loss')
            ax.set_title('Training History')
            ax.legend()
            ax.grid(True)
        
        curve_path = output_dir / 'training_curve.png'
        plt.tight_layout()
        plt.savefig(curve_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"\n✅ 학습 곡선 저장: {curve_path}")
    except Exception as e:
        print(f"⚠️ 학습 곡선 저장 실패: {e}")


if __name__ == '__main__':
    main()

