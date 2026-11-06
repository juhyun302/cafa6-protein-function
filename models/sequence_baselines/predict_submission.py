"""
CAFA 6 - Submission Prediction Script for Azure ML (GPU Accelerated)
=====================================================================
Test set 예측 및 submission.tsv 생성 스크립트

주요 기능:
- Test H5 임베딩 파일 로드
- 학습된 Attention MIL 모델로 예측 (GPU 가속)
- Label propagation (계층 구조 적용, GPU 가속)
- CAFA6 형식 submission.tsv 생성

최적화:
- Mixed Precision (FP16) 사용
- GPU Label Propagation
- 메모리 효율적인 일괄 처리
- Multi-worker I/O
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
from tqdm.auto import tqdm
import time
import h5py
from typing import Dict, List, Tuple

# 로컬 모듈 임포트
from attention_mil_model import AttentionMILClassifier, MILModelConfig
from hierarchical_cnn import load_ia_weights


# =========================================================================
# Test Chunk Embedding Dataset (H5 파일 기반) - Multi-worker 최적화
# =========================================================================

class TestChunkEmbeddingDataset(Dataset):
    """
    Test Chunk 임베딩 데이터셋 (H5 파일 기반) - Multi-worker 최적화

    h5_data_loader.py의 HDF5EmbeddingDataset 패턴 적용:
    - Lazy loading: 필요할 때만 HDF5 파일 로드
    - Multiprocessing 안전: pickle 직렬화 지원
    - 각 워커별 독립적인 파일 핸들 관리
    """

    def __init__(
        self,
        h5_path: Path
    ):
        """
        Args:
            h5_path: HDF5 파일 경로
        """
        self.h5_path = Path(h5_path)

        # 메타데이터 로드 (한 번만, 메모리에 캐시)
        with h5py.File(self.h5_path, 'r') as f:
            self.num_sequences = f.attrs.get('num_sequences', len(f['ids']))
            self.embedding_dim = f.attrs.get('embedding_dim', f['embeddings'].shape[1])

            # 전체 메타데이터를 메모리에 로드 (파일 접근 최소화)
            self.chunk_offsets = f['chunk_offsets'][:]
            self.chunk_counts = f['chunk_counts'][:]
            self.ids = f['ids'][:]

        # HDF5 파일 핸들 (lazy loading, 각 워커별로 별도 관리)
        self._h5_file = None

        print(f"✅ TestChunkEmbeddingDataset 초기화:")
        print(f"   시퀀스 수: {len(self.ids):,}")
        print(f"   임베딩 차원: {self.embedding_dim}")
        print(f"   최대 chunk 수: {int(self.chunk_counts.max())}")
        print(f"   평균 chunk/시퀀스: {self.chunk_counts.mean():.2f}")

    def reset_file_handle(self):
        """
        HDF5 파일 핸들 리셋 (멀티프로세싱용)
        각 워커에서 별도로 파일을 열기 위해 사용
        """
        if self._h5_file is not None:
            self._h5_file.close()
        self._h5_file = None

    def _get_h5_file(self):
        """HDF5 파일 핸들 반환 (lazy loading)"""
        if self._h5_file is None:
            self._h5_file = h5py.File(self.h5_path, 'r')
        return self._h5_file

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        """
        Returns:
            protein_id: 단백질 ID (str)
            embedding: (num_chunks, embedding_dim) - 원본 chunk 임베딩
            num_chunks: int - 실제 chunk 개수
        """
        # 워커별 파일 핸들 가져오기 (lazy loading)
        f = self._get_h5_file()

        # Chunk 임베딩 로드 (메모리 최적화: 필요한 부분만)
        offset = self.chunk_offsets[idx]
        count = self.chunk_counts[idx]

        # HDF5 슬라이싱 최적화 (연속 블록 읽기)
        chunks = np.asarray(f['embeddings'][offset:offset + count], dtype=np.float32)

        # 단백질 ID 디코딩
        protein_id = self.ids[idx]
        if isinstance(protein_id, bytes):
            protein_id = protein_id.decode('utf-8')

        # UniProt 형식 파싱: 'sp|Q6UXS9|CASPC_HUMAN' → 'Q6UXS9'
        if '|' in protein_id:
            parts = protein_id.split('|')
            if len(parts) >= 2:
                protein_id = parts[1]

        return {
            'protein_id': protein_id,
            'embedding': torch.from_numpy(chunks),
            'num_chunks': int(count)
        }

    def __del__(self):
        """파일 핸들 정리"""
        if self._h5_file is not None:
            self._h5_file.close()

    # =========================================================================
    # 멀티프로세싱 지원 (pickle 직렬화)
    # =========================================================================

    def __getstate__(self):
        """
        pickle 직렬화 시 호출됨
        파일 핸들은 프로세스 간 공유 불가능하므로 제외
        """
        state = self.__dict__.copy()
        state['_h5_file'] = None  # 파일 핸들 제외
        return state

    def __setstate__(self, state):
        """
        unpickle 후 호출됨 (각 워커 프로세스에서)
        파일 핸들은 None으로 설정하고, 나중에 lazy loading으로 열림
        """
        self.__dict__.update(state)
        self._h5_file = None  # 각 워커에서 _get_h5_file() 호출 시 새로 열림


def collate_fn_test(batch):
    """
    Test 데이터 collate 함수 (동적 패딩)
    """
    # 배치 내 최대 chunk 수
    max_chunks = max(item['num_chunks'] for item in batch)
    batch_size = len(batch)
    embedding_dim = batch[0]['embedding'].shape[1]

    # 패딩된 텐서 초기화
    embeddings = torch.zeros(batch_size, max_chunks, embedding_dim, dtype=torch.float32)
    attention_masks = torch.zeros(batch_size, max_chunks, dtype=torch.float32)
    protein_ids = []

    for i, item in enumerate(batch):
        num_chunks = item['num_chunks']
        embeddings[i, :num_chunks, :] = item['embedding']
        attention_masks[i, :num_chunks] = 1.0
        protein_ids.append(item['protein_id'])

    return {
        'protein_id': protein_ids,
        'embedding': embeddings,
        'attention_mask': attention_masks
    }


def worker_init_fn(worker_id):
    """
    DataLoader 워커 초기화 함수

    각 워커에서 HDF5 파일 핸들을 별도로 열기 위해 사용
    h5_data_loader.py의 패턴 적용
    """
    worker_info = torch.utils.data.get_worker_info()
    if worker_info is not None:
        dataset = worker_info.dataset
        # HDF5 파일 핸들 리셋 (각 워커에서 새로 열기)
        if hasattr(dataset, 'reset_file_handle'):
            dataset.reset_file_handle()
        # 각 워커별로 다른 random seed 설정
        np.random.seed(worker_info.seed % (2**32))


# =========================================================================
# Label Propagation (GPU 가속)
# =========================================================================

def propagate_predictions(
    predictions: np.ndarray,
    ancestor_matrix,
    batch_size: int = 512,
    use_amp: bool = True
) -> np.ndarray:
    """
    예측 결과에 label propagation 적용 (GPU 가속 + 메모리 최적화)

    각 GO term의 예측 확률을 조상(ancestor) GO term에 전파
    조상 term의 확률 = max(자신의 확률, 모든 자손 term의 확률)

    Args:
        predictions: (N_proteins, N_terms) 예측 확률
        ancestor_matrix: (N_terms, N_terms) 조상 관계 행렬 (sparse)
        batch_size: 배치 크기 (GPU에서는 큰 배치가 효율적)
        use_amp: Mixed Precision 사용 여부

    Returns:
        propagated: (N_proteins, N_terms) 전파된 예측 확률
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    use_gpu = torch.cuda.is_available()

    print(f"🔄 Label propagation 수행 중 (배치 크기: {batch_size}, GPU: {use_gpu}, AMP: {use_amp})...")

    # Ancestor matrix를 dense로 변환 (메모리 최적화: float16 사용)
    if hasattr(ancestor_matrix, 'toarray'):
        ancestor_dense = ancestor_matrix.toarray()
    else:
        ancestor_dense = ancestor_matrix

    # Transpose: ancestor_T[i, j] = 1 if term_j is ancestor of term_i
    # float16으로 변환하여 GPU 메모리 절약 (0/1 값이므로 정밀도 손실 없음)
    ancestor_T = torch.tensor(ancestor_dense.T, dtype=torch.float16, device=device)

    num_proteins = predictions.shape[0]
    result_batches = []

    # GPU 병렬 처리: 큰 배치로 한 번에 처리
    for start_idx in tqdm(range(0, num_proteins, batch_size), desc="GPU 전파"):
        end_idx = min(start_idx + batch_size, num_proteins)
        batch_preds = predictions[start_idx:end_idx]

        # float16으로 변환하여 GPU 메모리 절약 + 속도 향상
        batch_tensor = torch.tensor(batch_preds, dtype=torch.float16, device=device)

        # Mixed Precision 사용
        if use_amp and use_gpu:
            with torch.cuda.amp.autocast():
                # 전파: pred @ ancestor_T (GPU에서 병렬 행렬 곱셈)
                batch_propagated = torch.matmul(batch_tensor, ancestor_T)

                # Max pooling: 원본 예측과 전파된 값 중 최댓값
                batch_propagated = torch.maximum(batch_tensor, batch_propagated)
        else:
            batch_propagated = torch.matmul(batch_tensor, ancestor_T)
            batch_propagated = torch.maximum(batch_tensor, batch_propagated)

        # float32로 다시 변환하여 저장
        result_batches.append(batch_propagated.cpu().float().numpy())

        # 즉시 메모리 해제
        del batch_tensor, batch_propagated

        # 주기적으로 GPU 캐시 정리 (10 배치마다)
        if use_gpu and (start_idx // batch_size) % 10 == 0:
            torch.cuda.empty_cache()

    # 최종 GPU 캐시 정리
    del ancestor_T
    if use_gpu:
        torch.cuda.empty_cache()

    propagated = np.vstack(result_batches)

    print(f"✅ Label propagation 완료")
    return propagated


# =========================================================================
# Submission 생성 (GPU 가속 벡터화)
# =========================================================================

def create_submission_tsv_gpu_accelerated(
    protein_ids: List[str],
    predictions: np.ndarray,
    idx_to_term: Dict[int, str],
    output_path: Path,
    threshold: float = 0.01,
    batch_size: int = 1000
):
    """
    CAFA6 형식 submission.tsv 생성 (GPU 가속 + 벡터화)

    전략:
    1. GPU에서 torch.nonzero()로 필터링 (0.0001초)
    2. 유효한 데이터만 CPU로 전송 (메모리 효율)
    3. Pandas로 일괄 쓰기 (C 최적화)

    Args:
        protein_ids: 단백질 ID 리스트
        predictions: (N_proteins, N_terms) 예측 확률
        idx_to_term: 인덱스 -> GO term 매핑
        output_path: 출력 파일 경로
        threshold: 최소 확률 임계값
        batch_size: 배치 크기 (메모리 제약 시 줄일 것)
    """
    print(f"\n📝 Submission 파일 생성 중 (GPU 가속 벡터화)...")

    num_proteins, num_terms = predictions.shape
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    use_gpu = torch.cuda.is_available()

    # GO term 인덱스 -> 이름 매핑 (vectorized lookup용)
    go_term_names = np.array([idx_to_term[i] for i in range(num_terms)])

    all_data = []
    total_valid_predictions = 0

    # 배치 단위로 처리 (메모리 효율)
    for start_idx in tqdm(range(0, num_proteins, batch_size), desc="GPU 벡터화"):
        end_idx = min(start_idx + batch_size, num_proteins)
        batch_preds = predictions[start_idx:end_idx]
        batch_ids = protein_ids[start_idx:end_idx]

        # GPU로 전송
        batch_tensor = torch.tensor(batch_preds, dtype=torch.float32, device=device)

        # ★ GPU에서 즉시 필터링: torch.nonzero() (초고속!)
        if use_gpu:
            valid_mask = batch_tensor >= threshold
            valid_indices = torch.nonzero(valid_mask, as_tuple=False)  # (N_valid, 2)

            # GPU에서 값 추출
            valid_probs = batch_tensor[valid_mask]

            # CPU로 전송 (유효한 데이터만!)
            protein_indices = valid_indices[:, 0].cpu().numpy()
            term_indices = valid_indices[:, 1].cpu().numpy()
            probs = valid_probs.cpu().numpy()
        else:
            # CPU 폴백
            valid_mask = batch_preds >= threshold
            protein_indices, term_indices = np.where(valid_mask)
            probs = batch_preds[valid_mask]

        del batch_tensor
        if use_gpu:
            torch.cuda.empty_cache()

        # ★ 벡터화된 데이터 생성 (Python for loop 없음!)
        batch_protein_ids = np.array(batch_ids)[protein_indices]
        batch_go_terms = go_term_names[term_indices]

        # DataFrame으로 일괄 변환
        batch_df = pd.DataFrame({
            'protein_id': batch_protein_ids,
            'go_term': batch_go_terms,
            'confidence': probs
        })

        all_data.append(batch_df)
        total_valid_predictions += len(batch_df)

    # ★ Pandas 고속 쓰기 (C 최적화, file.write보다 10배 빠름)
    print(f"💾 파일 쓰기 중 ({total_valid_predictions:,}개 예측)...")

    if all_data:
        final_df = pd.concat(all_data, ignore_index=True)

        # 단백질별로 확률 높은 순 정렬 (선택적, 필요 시)
        # final_df = final_df.sort_values(['protein_id', 'confidence'],
        #                                   ascending=[True, False])

        # TSV 형식으로 저장 (헤더 없음)
        final_df.to_csv(
            output_path,
            sep='\t',
            header=False,
            index=False,
            float_format='%.4f'
        )

    print(f"✅ Submission 파일 저장: {output_path}")

    # 통계 출력
    avg_predictions_per_protein = total_valid_predictions / num_proteins

    print(f"\n📊 Submission 통계:")
    print(f"   총 단백질 수: {num_proteins:,}")
    print(f"   총 예측 수 (≥{threshold}): {total_valid_predictions:,}")
    print(f"   단백질당 평균 예측: {avg_predictions_per_protein:.1f}")


def create_submission_tsv(
    protein_ids: List[str],
    predictions: np.ndarray,
    idx_to_term: Dict[int, str],
    output_path: Path,
    threshold: float = 0.01,
    batch_size: int = 1000
):
    """
    CAFA6 형식 submission.tsv 생성 (GPU 가속 래퍼)

    Args:
        protein_ids: 단백질 ID 리스트
        predictions: (N_proteins, N_terms) 예측 확률
        idx_to_term: 인덱스 -> GO term 매핑
        output_path: 출력 파일 경로
        threshold: 최소 확률 임계값
        batch_size: TSV 생성 배치 크기
    """
    # GPU 가속 버전 사용
    create_submission_tsv_gpu_accelerated(
        protein_ids=protein_ids,
        predictions=predictions,
        idx_to_term=idx_to_term,
        output_path=output_path,
        threshold=threshold,
        batch_size=batch_size
    )


# =========================================================================
# 메인 함수
# =========================================================================

def parse_args():
    """커맨드라인 인자 파싱"""
    parser = argparse.ArgumentParser(description='CAFA 6 - Submission Prediction (GPU Accelerated)')

    # 데이터 경로
    parser.add_argument('--data_path', type=str, required=True,
                        help='Azure ML 데이터셋 마운트 경로')
    parser.add_argument('--test_h5', type=str,
                        default='embedded_fasta_esm2_3b/test/esm2_3b_chunk_embeddings_224309sq.h5',
                        help='Test H5 임베딩 파일 경로 (data_path 기준 상대 경로)')

    # 모델 경로
    parser.add_argument('--model_path', type=str, required=True,
                        help='학습된 모델 체크포인트 경로 (.pth)')

    # GO 매핑 및 계층 구조
    parser.add_argument('--go_mappings', type=str,
                        default='go_basic/go_term_mappings.pkl',
                        help='GO term 매핑 파일 경로')
    parser.add_argument('--ancestor_matrix', type=str,
                        default='go_basic/go_ancestor_matrix.npz',
                        help='GO ancestor matrix 파일 경로')

    # 출력
    parser.add_argument('--output_dir', type=str, default='./outputs',
                        help='출력 디렉토리')
    parser.add_argument('--output_filename', type=str, default='submission.tsv',
                        help='출력 파일명')
    parser.add_argument('--threshold', type=float, default=0.01,
                        help='최소 확률 임계값 (이 값 이상만 출력)')

    # 예측 설정
    parser.add_argument('--batch_size', type=int, default=128,
                        help='배치 크기 (GPU 활용 최대화)')
    parser.add_argument('--num_workers', type=int, default=6,
                        help='DataLoader worker 수')
    parser.add_argument('--propagation_batch_size', type=int, default=1024,
                        help='Label propagation 배치 크기 (GPU에서는 큰 배치가 효율적)')
    parser.add_argument('--tsv_batch_size', type=int, default=1000,
                        help='TSV 생성 배치 크기 (GPU 벡터화용)')

    # GPU 최적화 설정
    parser.add_argument('--use_amp', action='store_true', default=True,
                        help='Mixed Precision (FP16) 사용 (2배 속도 향상)')
    parser.add_argument('--compile_model', action='store_true', default=False,
                        help='torch.compile() 사용 (PyTorch 2.0+, 추가 최적화)')

    # 하위 호환성을 위한 더미 인자
    parser.add_argument('--buffer_size', type=int, default=1000,
                        help='(사용하지 않음, 하위 호환성용)')
    parser.add_argument('--gpu_propagation', action='store_true', default=True,
                        help='(사용하지 않음, 하위 호환성용)')

    args = parser.parse_args()
    return args


def main():
    """메인 예측 함수"""
    args = parse_args()

    # 출력 디렉토리 생성
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n{'='*70}")
    print(f"🚀 CAFA 6 - Submission Prediction (GPU Accelerated)")
    print(f"{'='*70}")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Mixed Precision (AMP): {args.use_amp}")

    data_path = Path(args.data_path)

    # =========================================================================
    # 1. GO Term 매핑 및 계층 구조 로드
    # =========================================================================
    print("\n" + "=" * 70)
    print("📊 GO Term 매핑 로드")
    print("=" * 70)

    go_mappings_path = data_path / args.go_mappings
    with open(go_mappings_path, 'rb') as f:
        go_mappings = pickle.load(f)

    term_to_idx = go_mappings['go_to_idx']
    idx_to_term = go_mappings['idx_to_go']
    num_go_terms = len(term_to_idx)

    print(f"✅ GO Term 총 개수: {num_go_terms:,}")

    # Ancestor matrix 로드
    ancestor_matrix_path = data_path / args.ancestor_matrix
    ancestor_matrix = load_npz(ancestor_matrix_path)
    print(f"✅ GO ancestor matrix shape: {ancestor_matrix.shape}")

    # =========================================================================
    # 2. 학습된 모델 로드
    # =========================================================================
    print("\n" + "=" * 70)
    print("🏗️ 학습된 모델 로드")
    print("=" * 70)

    checkpoint = torch.load(args.model_path, map_location=device, weights_only=False)

    # 모델 설정 복원
    model_config = MILModelConfig.from_dict(checkpoint['model_config'])
    print(f"✅ 모델 설정:")
    print(f"   Embedding dim: {model_config.embedding_dim}")
    print(f"   Num classes: {model_config.num_classes}")
    print(f"   MIL hidden dim: {model_config.mil_hidden_dim}")
    print(f"   Gated attention: {model_config.use_gated_attention}")

    # 모델 생성 및 가중치 로드
    model = AttentionMILClassifier.from_config(model_config).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    print(f"✅ 모델 로드 완료")
    print(f"   Epoch: {checkpoint['epoch']}")
    print(f"   Best Val Loss: {checkpoint.get('best_val_loss', 'N/A')}")
    if 'best_val_fmax' in checkpoint:
        print(f"   Best Val F-max: {checkpoint['best_val_fmax']:.4f}")

    # torch.compile 적용 (선택적)
    if args.compile_model and hasattr(torch, 'compile'):
        print(f"🚀 torch.compile() 적용 중... (첫 배치는 느릴 수 있음)")
        try:
            model = torch.compile(model, mode='max-autotune')
            print(f"✅ torch.compile() 적용 완료")
        except Exception as e:
            print(f"⚠️ torch.compile() 실패 (무시하고 계속): {e}")

    # =========================================================================
    # 3. Test 데이터셋 및 DataLoader 생성
    # =========================================================================
    print("\n" + "=" * 70)
    print("📂 Test 데이터셋 준비")
    print("=" * 70)

    test_h5_path = data_path / args.test_h5
    print(f"Test H5: {test_h5_path}")

    if not test_h5_path.exists():
        raise FileNotFoundError(f"Test H5 파일을 찾을 수 없습니다: {test_h5_path}")

    test_dataset = TestChunkEmbeddingDataset(h5_path=test_h5_path)

    use_cuda = torch.cuda.is_available()
    pin_memory = use_cuda

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn_test,
        worker_init_fn=worker_init_fn if args.num_workers > 0 else None,
        persistent_workers=True if args.num_workers > 0 else False,
        prefetch_factor=2 if args.num_workers > 0 else None
    )

    print(f"✅ Test 시퀀스 수: {len(test_dataset):,}")
    print(f"✅ 배치 크기: {args.batch_size}")
    print(f"✅ Num workers: {args.num_workers}")

    # =========================================================================
    # 4. 예측 수행 (GPU 가속 + 메모리 효율)
    # =========================================================================
    print("\n" + "=" * 70)
    print("🔮 예측 수행 중 (GPU 가속)...")
    print("=" * 70)

    all_protein_ids = []
    all_predictions = []

    # Mixed Precision 활성화
    use_amp = args.use_amp and torch.cuda.is_available()

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(test_loader, desc="GPU 예측")):
            # Non-blocking transfer로 CPU→GPU 전송 오버헤드 최소화
            embeddings = batch['embedding'].to(device, non_blocking=True)
            attention_mask = batch['attention_mask'].to(device, non_blocking=True)
            protein_ids = batch['protein_id']

            # Mixed Precision (FP16) 예측
            if use_amp:
                with torch.cuda.amp.autocast():
                    logits = model(embeddings, attention_mask)
            else:
                logits = model(embeddings, attention_mask)

            # Sigmoid를 GPU에서 계산 후 CPU로 전송
            probs = torch.sigmoid(logits)

            all_protein_ids.extend(protein_ids)
            all_predictions.append(probs.cpu().numpy())

            # 즉시 메모리 해제
            del embeddings, attention_mask, logits, probs

            # 주기적으로 GPU 캐시 정리 (10 배치마다)
            if batch_idx % 10 == 0:
                torch.cuda.empty_cache()

    # 최종 GPU 캐시 정리
    torch.cuda.empty_cache()

    # 예측 결과 병합
    predictions = np.vstack(all_predictions)  # (N_proteins, N_terms)

    print(f"✅ 예측 완료:")
    print(f"   단백질 수: {len(all_protein_ids):,}")
    print(f"   예측 shape: {predictions.shape}")
    print(f"   예측 범위: [{predictions.min():.4f}, {predictions.max():.4f}]")

    # =========================================================================
    # 5. Submission 파일 생성 (Raw - 전파 없음)
    # =========================================================================
    print("\n" + "=" * 70)
    print("📝 Submission 파일 생성 (Raw - 후처리 없음)")
    print("=" * 70)

    output_raw = output_dir / 'submission_raw.tsv'
    create_submission_tsv(
        protein_ids=all_protein_ids,
        predictions=predictions,
        idx_to_term=idx_to_term,
        output_path=output_raw,
        threshold=args.threshold,
        batch_size=args.tsv_batch_size
    )

    print(f"✅ Raw submission 저장: {output_raw}")

    # =========================================================================
    # 6. Label Propagation (GPU 가속)
    # =========================================================================
    print("\n" + "=" * 70)
    print("🔄 Label Propagation (GPU 가속)")
    print("=" * 70)

    predictions_propagated = propagate_predictions(
        predictions=predictions,
        ancestor_matrix=ancestor_matrix,
        batch_size=args.propagation_batch_size,
        use_amp=use_amp
    )

    print(f"✅ 전파 후 예측 범위: [{predictions_propagated.min():.4f}, {predictions_propagated.max():.4f}]")

    # =========================================================================
    # 7. Submission 파일 생성 (Propagated - 전파 적용)
    # =========================================================================
    print("\n" + "=" * 70)
    print("📝 Submission 파일 생성 (Propagated - 후처리 적용)")
    print("=" * 70)

    output_propagated = output_dir / 'submission_propagated.tsv'
    create_submission_tsv(
        protein_ids=all_protein_ids,
        predictions=predictions_propagated,
        idx_to_term=idx_to_term,
        output_path=output_propagated,
        threshold=args.threshold,
        batch_size=args.tsv_batch_size
    )

    print(f"✅ Propagated submission 저장: {output_propagated}")

    print("\n" + "=" * 70)
    print("✅ 예측 완료!")
    print("=" * 70)
    print(f"📄 Raw submission: {output_raw}")
    print(f"📄 Propagated submission: {output_propagated}")


if __name__ == '__main__':
    main()
