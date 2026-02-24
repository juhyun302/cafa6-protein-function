"""
CAFA 6 - 1D CNN Model Training Script for Azure ML
===================================================
Azure ML Studio에서 컴퓨팅 클러스터를 사용한 분산 학습 스크립트

주요 기능:
- Azure ML 데이터셋 자동 마운트
- MLflow 자동 로깅 (metrics, parameters, model)
- Checkpoint 자동 저장 및 복구
- 분산 학습 지원 (선택적)
- Early Stopping & Learning Rate Scheduling
"""

import os
import sys
import argparse
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from scipy.sparse import load_npz
from pathlib import Path
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm
import time
import json

# Azure ML & MLflow
import mlflow
import mlflow.pytorch

# 로컬 모듈 임포트 (src 폴더가 푸시됨)
from h5_data_loder import (
    H5MetadataAnalyzer,
    HDF5EmbeddingDataset,
    worker_init_fn,
    collate_fn_train
)
from model import (
    Hierarchical1DCNNClassifier,
    ModelConfig,
    TrainingConfig,
    train_epoch,
    validate_epoch,
    AsymmetricLoss,
    load_ia_weights,
    create_optimizer,
    create_scheduler,
    create_scaler,
)


def parse_args():
    """커맨드라인 인자 파싱"""
    parser = argparse.ArgumentParser(description='CAFA 6 - 1D CNN Training on Azure ML')
    
    # ==================== Azure ML 데이터셋 ====================
    parser.add_argument('--data_path', type=str, required=True,
                        help='Azure ML 데이터셋 마운트 경로')
    
    # ==================== 데이터 경로 (데이터셋 내 상대 경로) ====================
    parser.add_argument('--dataset_name', type=str, 
                        default='CAFA6_82404sq_Asymmetric Loss',
                        help='데이터셋 폴더 이름')
    
    # ==================== 모델 아키텍처 ====================
    parser.add_argument('--embedding_dim', type=int, default=1280,
                        help='ESM2 임베딩 차원')
    parser.add_argument('--max_seq_length', type=int, default=1024,
                        help='최대 시퀀스 길이')
    parser.add_argument('--conv_channels', type=str, default='512,1024,2048',
                        help='Conv 레이어 채널 수 (comma-separated)')
    parser.add_argument('--kernel_sizes', type=str, default='7,5,3',
                        help='Conv 커널 크기 (comma-separated)')
    parser.add_argument('--fc_dims', type=str, default='2048',
                        help='FC 레이어 차원 (comma-separated)')
    parser.add_argument('--dropout', type=float, default=0.3,
                        help='Dropout 비율')
    parser.add_argument('--conv_dropout_ratio', type=float, default=0.5,
                        help='Conv Dropout 비율')
    parser.add_argument('--use_residual', action='store_true', default=True,
                        help='Residual Connection 사용')
    parser.add_argument('--pooling_mode', type=str, default='concat',
                        choices=['concat', 'max', 'avg'],
                        help='Global Pooling 방식')
    
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
    
    # 리스트 파싱
    args.conv_channels = [int(x) for x in args.conv_channels.split(',')]
    args.kernel_sizes = [int(x) for x in args.kernel_sizes.split(',')]
    args.fc_dims = [int(x) for x in args.fc_dims.split(',')]
    
    return args


def setup_paths(args):
    """Azure ML 데이터셋 경로 설정"""
    # Azure ML 데이터셋 마운트 경로 처리
    data_path = Path(args.data_path)
    
    print("=" * 70)
    print("📂 데이터셋 경로 검증")
    print("=" * 70)
    print(f"🔍 data_path: {data_path}")
    print(f"🔍 dataset_name: {args.dataset_name}")
    print(f"🔍 data_path exists: {data_path.exists()}")
    
    # 실제 폴더 구조 확인 (디버깅용)
    if data_path.exists():
        print(f"\n📁 data_path 내용 (상위 10개):")
        try:
            items = list(data_path.iterdir())[:10]
            for item in sorted(items):
                item_type = "📁" if item.is_dir() else "📄"
                print(f"   {item_type} {item.name}")
        except Exception as e:
            print(f"   ⚠️ 폴더 읽기 실패: {e}")
    
    # 데이터셋 루트 경로 결정 전략:
    # 1. data_path에 예상 폴더가 있는지 확인 (embedded_fasta_for_1dcnn, go_basic, cafa-6-protein-function-prediction)
    # 2. 있으면 data_path가 이미 데이터셋 루트
    # 3. 없으면 data_path/dataset_name이 데이터셋 루트
    
    expected_folders = ['embedded_fasta_for_1dcnn', 'go_basic', 'cafa-6-protein-function-prediction']
    has_expected_structure = any((data_path / folder).exists() for folder in expected_folders)
    
    if has_expected_structure:
        # data_path가 이미 데이터셋 루트
        base_dir = data_path
        print(f"\n✅ 감지: data_path가 이미 데이터셋 루트")
    else:
        # data_path/dataset_name이 데이터셋 루트일 가능성
        potential_base = data_path / args.dataset_name
        if potential_base.exists() and any((potential_base / folder).exists() for folder in expected_folders):
            base_dir = potential_base
            print(f"\n✅ 감지: {args.dataset_name} 하위 폴더가 데이터셋 루트")
        else:
            # 기본값: data_path 사용 (에러는 나중에 발생)
            base_dir = data_path
            print(f"\n⚠️ 예상 폴더 구조를 찾지 못함, data_path 사용")
    
    print(f"🔍 최종 base_dir: {base_dir}")
    
    paths = {
        'base_dir': base_dir,
        'train_h5': base_dir / 'embedded_fasta_for_1dcnn' / 'train' / 'train_embeddings_82404sq.h5',
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
        if not exists and name not in ['go_dir']:  # go_dir는 선택적
            raise FileNotFoundError(f"필수 파일 없음: {path}")
    
    return paths


def load_go_mappings(paths):
    """GO Term 매핑 및 계층 구조 로드"""
    print("\n" + "=" * 70)
    print("📊 GO Term 매핑 로드")
    print("=" * 70)
    
    # GO Term 매핑
    with open(paths['go_mappings'], 'rb') as f:
        go_mappings = pickle.load(f)
    
    term_to_idx = go_mappings['go_to_idx']
    idx_to_term = go_mappings['idx_to_go']
    num_go_terms = len(term_to_idx)
    
    print(f"✅ GO Term 총 개수: {num_go_terms:,}")
    
    # GO 계층 구조
    ancestor_matrix = load_npz(paths['ancestor_matrix'])
    print(f"✅ GO ancestor matrix shape: {ancestor_matrix.shape}")
    
    return term_to_idx, idx_to_term, num_go_terms, ancestor_matrix


def load_training_labels(paths, train_ids, term_to_idx, num_go_terms, ancestor_matrix):
    """학습 라벨 로드 및 전파"""
    print("\n" + "=" * 70)
    print("🏷️ 학습 라벨 로드")
    print("=" * 70)
    
    # train_terms.tsv 로드
    train_terms_df = pd.read_csv(paths['train_terms'], sep='\t')
    train_terms_df.columns = ['EntryID', 'GO_Term', 'Aspect']
    
    print(f"✅ Train terms 총 개수: {len(train_terms_df):,}")
    print(f"✅ 고유 단백질: {train_terms_df['EntryID'].nunique():,}")
    print(f"✅ 고유 GO Terms: {train_terms_df['GO_Term'].nunique():,}")
    
    # 단백질 ID 매핑
    protein_id_to_idx = {protein_id: idx for idx, protein_id in enumerate(train_ids)}
    train_terms_df = train_terms_df[train_terms_df['EntryID'].isin(set(train_ids))].copy()
    
    # Multi-label binary matrix 생성
    num_proteins = len(train_ids)
    label_matrix = np.zeros((num_proteins, num_go_terms), dtype=np.float32)
    
    for _, row in tqdm(train_terms_df.iterrows(), total=len(train_terms_df), desc="라벨 매트릭스 구축"):
        protein_id = row['EntryID']
        go_term = row['GO_Term']
        
        if protein_id in protein_id_to_idx and go_term in term_to_idx:
            protein_idx = protein_id_to_idx[protein_id]
            go_idx = term_to_idx[go_term]
            label_matrix[protein_idx, go_idx] = 1.0
    
    print(f"✅ Label matrix shape: {label_matrix.shape}")
    print(f"✅ Positive labels: {label_matrix.sum():.0f}")
    print(f"✅ Sparsity: {1 - label_matrix.sum() / label_matrix.size:.4%}")
    
    # 라벨 전파 (GPU 가속)
    print("\n🔄 라벨 전파 (계층 구조 적용)...")
    label_matrix_propagated = propagate_labels_gpu(label_matrix, ancestor_matrix)
    print(f"✅ 전파 후 positive labels: {label_matrix_propagated.sum():.0f}")
    print(f"✅ 추가된 라벨: {label_matrix_propagated.sum() - label_matrix.sum():.0f}")
    
    return label_matrix_propagated


def propagate_labels_gpu(label_matrix, ancestor_matrix, batch_size=4096):
    """GPU 가속 라벨 전파 (배치 처리)"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Ancestor matrix를 GPU로 (전치)
    if hasattr(ancestor_matrix, 'toarray'):
        ancestor_T_dense = ancestor_matrix.T.toarray()
    else:
        ancestor_T_dense = ancestor_matrix.T
    
    ancestor_tensor_T = torch.tensor(ancestor_T_dense, dtype=torch.float32).to(device)
    
    num_proteins = label_matrix.shape[0]
    result_batches = []
    
    for start_idx in tqdm(range(0, num_proteins, batch_size), desc="라벨 전파"):
        end_idx = min(start_idx + batch_size, num_proteins)
        
        # 배치 슬라이스
        batch_input = label_matrix[start_idx:end_idx]
        batch_tensor = torch.tensor(batch_input, dtype=torch.float32).to(device)
        
        # 행렬 곱 (배치 x 계층 구조)
        batch_result = torch.matmul(batch_tensor, ancestor_tensor_T)
        batch_result = (batch_result > 0).float()
        
        result_batches.append(batch_result.cpu().numpy())
        
        del batch_tensor, batch_result
    
    torch.cuda.empty_cache()
    return np.vstack(result_batches)


def create_dataloaders(args, train_h5_path, train_offsets, train_lengths, 
                       label_matrix, max_seq_length):
    """Train/Validation DataLoader 생성"""
    print("\n" + "=" * 70)
    print("🔄 DataLoader 생성")
    print("=" * 70)
    
    # Train/Val split
    num_samples = len(train_offsets)
    train_idx, val_idx = train_test_split(
        np.arange(num_samples),
        test_size=args.val_split,
        random_state=args.random_seed
    )
    
    y_train = label_matrix[train_idx]
    y_val = label_matrix[val_idx]
    
    print(f"✅ Train: {len(train_idx):,} samples")
    print(f"✅ Validation: {len(val_idx):,} samples")
    
    # 데이터셋 생성
    train_dataset = HDF5EmbeddingDataset(
        h5_path=train_h5_path,
        indices=train_idx,
        labels=y_train,
        offsets=train_offsets,
        lengths=train_lengths,
        max_length=max_seq_length,
        pad=False
    )
    
    val_dataset = HDF5EmbeddingDataset(
        h5_path=train_h5_path,
        indices=val_idx,
        labels=y_val,
        offsets=train_offsets,
        lengths=train_lengths,
        max_length=max_seq_length,
        pad=False
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
        worker_init_fn=worker_init_fn if args.num_workers > 0 else None,
        collate_fn=collate_fn_train,
        persistent_workers=True if args.num_workers > 0 else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        worker_init_fn=worker_init_fn if args.num_workers > 0 else None,
        collate_fn=collate_fn_train,
        persistent_workers=True if args.num_workers > 0 else False
    )
    
    print(f"✅ Batch size: {args.batch_size}")
    print(f"✅ Num workers: {args.num_workers}")
    print(f"✅ Pin memory: {pin_memory}")
    
    return train_loader, val_loader


def setup_mlflow(args):
    """MLflow 실험 설정"""
    print("\n" + "=" * 70)
    print("📊 MLflow 설정")
    print("=" * 70)
    
    try:
        # Azure ML은 자동으로 MLflow tracking URI 설정
        mlflow.set_experiment("CAFA6_1D_CNN_Training")
        
        # 하이퍼파라미터 로깅
        mlflow.log_params({
            'embedding_dim': args.embedding_dim,
            'max_seq_length': args.max_seq_length,
            'conv_channels': str(args.conv_channels),
            'kernel_sizes': str(args.kernel_sizes),
            'fc_dims': str(args.fc_dims),
            'dropout': args.dropout,
            'conv_dropout_ratio': args.conv_dropout_ratio,
            'use_residual': args.use_residual,
            'pooling_mode': args.pooling_mode,
            'batch_size': args.batch_size,
            'num_epochs': args.num_epochs,
            'learning_rate': args.learning_rate,
            'weight_decay': args.weight_decay,
            'grad_clip_norm': args.grad_clip_norm,
            'use_ic_weighted_loss': args.use_ic_weighted_loss,
            'gamma_pos': args.gamma_pos,
            'gamma_neg': args.gamma_neg,
            'asl_clip': args.asl_clip,
            'scheduler_factor': args.scheduler_factor,
            'scheduler_patience': args.scheduler_patience,
            'early_stop_patience': args.early_stop_patience,
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
    # ==================== 인자 파싱 ====================
    args = parse_args()
    
    # 출력 디렉토리 생성
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Device 설정
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n{'='*70}")
    print(f"🚀 CAFA 6 - 1D CNN Training on Azure ML")
    print(f"{'='*70}")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")
    print(f"PyTorch Version: {torch.__version__}")
    
    # ==================== MLflow 설정 ====================
    setup_mlflow(args)
    
    # ==================== 경로 설정 ====================
    paths = setup_paths(args)
    
    # ==================== GO 매핑 로드 ====================
    term_to_idx, idx_to_term, num_go_terms, ancestor_matrix = load_go_mappings(paths)
    
    # ==================== 임베딩 메타데이터 로드 ====================
    print("\n" + "=" * 70)
    print("📥 임베딩 메타데이터 로드")
    print("=" * 70)
    
    train_analyzer = H5MetadataAnalyzer(paths['train_h5'])
    train_metadata = train_analyzer.metadata
    train_ids = train_analyzer.ids
    train_offsets = train_analyzer.offsets
    train_lengths = train_analyzer.lengths
    
    print(f"✅ Train 시퀀스 수: {train_analyzer.num_sequences:,}")
    print(f"✅ Train 총 토큰 수: {train_analyzer.total_tokens:,}")
    
    # ==================== 학습 라벨 로드 ====================
    label_matrix = load_training_labels(
        paths, train_ids, term_to_idx, num_go_terms, ancestor_matrix
    )
    
    # ==================== DataLoader 생성 ====================
    train_loader, val_loader = create_dataloaders(
        args, paths['train_h5'], train_offsets, train_lengths,
        label_matrix, args.max_seq_length
    )
    
    # ==================== 모델 생성 ====================
    print("\n" + "=" * 70)
    print("🏗️ 모델 생성")
    print("=" * 70)
    
    model_config = ModelConfig(
        embedding_dim=args.embedding_dim,
        num_classes=num_go_terms,
        conv_channels=args.conv_channels,
        kernel_sizes=args.kernel_sizes,
        fc_dims=args.fc_dims,
        dropout=args.dropout,
        conv_dropout_ratio=args.conv_dropout_ratio,
        use_residual=args.use_residual,
        pooling_mode=args.pooling_mode,
        use_batch_norm=True,
        activation='relu'
    )
    
    model = Hierarchical1DCNNClassifier.from_config(model_config).to(device)
    print(model.summary())
    
    # 모델 파라미터 수 로깅
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    try:
        mlflow.log_param('total_parameters', total_params)
        mlflow.log_param('trainable_parameters', trainable_params)
    except Exception as e:
        print(f"⚠️ MLflow 파라미터 로깅 실패: {e}")
    
    # ==================== 손실 함수 ====================
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
        print(f"   gamma_pos={args.gamma_pos}, gamma_neg={args.gamma_neg}")
    else:
        criterion = nn.BCEWithLogitsLoss()
        print(f"✅ BCEWithLogitsLoss")
    
    # ==================== 학습 설정 ====================
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
    
    # Optimizer, Scheduler, Scaler
    optimizer = create_optimizer(model, config=training_config, optimizer_type='adamw')
    scheduler = create_scheduler(optimizer, config=training_config, scheduler_type=args.scheduler_type)
    scaler = create_scaler(enabled=args.use_amp)
    
    print(f"✅ Optimizer: AdamW (lr={args.learning_rate}, wd={args.weight_decay})")
    if args.scheduler_type == 'cosine_warm_restarts':
        print(f"✅ Scheduler: CosineAnnealingWarmRestarts (T_0={args.cosine_t_0}, T_mult={args.cosine_t_mult}, min_lr={args.scheduler_min_lr})")
    else:
        print(f"✅ Scheduler: {args.scheduler_type} (factor={args.scheduler_factor}, patience={args.scheduler_patience})")
    print(f"✅ AMP: {args.use_amp}")
    
    # ==================== 학습 루프 ====================
    print("\n" + "=" * 70)
    print("🚀 학습 시작")
    print("=" * 70)
    
    best_val_loss = float('inf')
    patience_counter = 0
    train_losses = []
    val_losses = []
    
    for epoch in range(args.num_epochs):
        epoch_start_time = time.time()
        print(f"\n{'='*70}")
        print(f"Epoch {epoch+1}/{args.num_epochs}")
        print(f"{'='*70}")
        
        # 학습
        train_loss = train_epoch(
            model, train_loader, optimizer, criterion, device,
            scaler=scaler, config=training_config
        )
        train_losses.append(train_loss)
        
        # 검증
        val_loss = validate_epoch(
            model, val_loader, criterion, device,
            config=training_config
        )
        val_losses.append(val_loss)
        
        # Learning rate scheduler 업데이트
        if args.scheduler_type == 'reduce_on_plateau':
            scheduler.step(val_loss)
        else:
            # cosine, step, cosine_warm_restarts는 에폭 기반
            scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        
        epoch_time = time.time() - epoch_start_time
        
        # 로깅
        print(f"\n📊 Epoch {epoch+1} 결과:")
        print(f"   Train Loss: {train_loss:.6f}")
        print(f"   Val Loss: {val_loss:.6f}")
        print(f"   LR: {current_lr:.2e}")
        try:
            mlflow.log_metrics({
                'train_loss': train_loss,
                'val_loss': val_loss,
                'learning_rate': current_lr,
                'epoch_time': epoch_time
            }, step=epoch)
        except Exception as e:
            print(f"   ⚠️ MLflow 메트릭 로깅 실패: {e}")
        
        # 모델 저장 (개선 시)
        if val_loss < best_val_loss * (1 - args.early_stop_min_delta):
            improvement = best_val_loss - val_loss
            best_val_loss = val_loss
            
            # Outputs 폴더 생성 (없으면)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Checkpoint 저장
            checkpoint_path = output_dir / 'best_model.pth'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'model_config': model_config.to_dict(),
                'training_config': training_config.to_dict(),
                'args': vars(args),
            }, checkpoint_path)
            
            print(f"   ✅ Best model saved: {checkpoint_path}")
            print(f"   ✅ Val Loss: {val_loss:.6f} (improvement: {improvement:.6f})")
            patience_counter = 0
        else:
            patience_counter += 1
            print(f"   ⏳ No improvement ({patience_counter}/{args.early_stop_patience})")
        
        # Early stopping
        if patience_counter >= args.early_stop_patience:
            print(f"\n⚠️ Early stopping at epoch {epoch+1}")
            try:
                mlflow.log_metric('early_stopped_epoch', epoch)
            except Exception as e:
                print(f"⚠️ MLflow 메트릭 로깅 실패: {e}")
            break
    
    # ==================== 학습 완료 ====================
    print("\n" + "=" * 70)
    print("✅ 학습 완료!")
    print("=" * 70)
    print(f"Best validation loss: {best_val_loss:.6f}")
    print(f"Total epochs trained: {len(train_losses)}")
    print(f"모델 저장 경로: {output_dir / 'best_model.pth'}")
    
    # 최종 메트릭 로깅
    try:
        mlflow.log_metrics({
            'best_val_loss': best_val_loss,
            'total_epochs': len(train_losses),
        })
    except Exception as e:
        print(f"⚠️ MLflow 최종 메트릭 로깅 실패: {e}")
    
    # 학습 곡선 저장
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training History')
    plt.legend()
    plt.grid(True)
    
    # Outputs 폴더에 저장
    output_dir.mkdir(parents=True, exist_ok=True)
    curve_path = output_dir / 'training_curve.png'
    plt.savefig(curve_path)
    plt.close()
    
    print(f"\n✅ 학습 곡선 저장: {curve_path}")


if __name__ == '__main__':
    main()