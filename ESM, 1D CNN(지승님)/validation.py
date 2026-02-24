"""
CAFA6 Validation Module (Azure ML Studio Version)
==================================================
학습된 모델의 성능을 Validation Set으로 평가하는 모듈 (Azure ML용)

주요 기능:
- Azure ML 데이터셋 자동 마운트
- Label Matrix 생성 및 계층 구조 전파
- CAFA6 Weighted F-max & S-min 계산
- Sklearn 메트릭 평가 (다양한 임계값)
- MLflow 로깅 (메트릭, 아티팩트)
- 전처리된 데이터 캐싱 (메모리 효율화)

Azure ML 실행:
    python validation.py \
        --model_path /path/to/model.pth \
        --output_dir ./outputs

Local 실행:
    python validation.py \
        --model_path ./best_model.pth \
        --output_dir ./outputs \
        --local_mode
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import pickle
import torch
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union
from tqdm.auto import tqdm
from sklearn.metrics import (
    f1_score, precision_score, recall_score,
    hamming_loss, average_precision_score
)
from scipy import sparse
from scipy.sparse import csr_matrix
import pickle
import hashlib
import json


class ValidationEvaluator:
    """
    CAFA6 Validation 평가 전담 클래스
    
    Attributes:
        train_ids: 학습 데이터의 단백질 ID 리스트
        term_to_idx: GO term -> index 매핑 딕셔너리
        num_go_terms: 전체 GO term 개수
        ic_weights_np: IC 가중치 배열 (N_terms,)
        cache_dir: 전처리 데이터 캐시 디렉토리
        device: PyTorch 디바이스 (GPU/CPU)
    """
    
    def __init__(
        self,
        train_ids: List[str],
        term_to_idx: Dict[str, int],
        ic_weights_np: np.ndarray,
        cache_dir: Union[str, Path] = "./validation_init",
        device: Optional[torch.device] = None
    ):
        """
        Args:
            train_ids: 학습 데이터의 단백질 ID 리스트
            term_to_idx: GO term -> index 매핑 딕셔너리
            ic_weights_np: IC 가중치 배열 (N_terms,)
            cache_dir: 캐시 디렉토리 경로
            device: PyTorch 디바이스 (None이면 자동 선택)
        """
        self.train_ids = train_ids
        self.term_to_idx = term_to_idx
        self.num_go_terms = len(term_to_idx)
        self.ic_weights_np = ic_weights_np
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
        
        # 단백질 ID -> 인덱스 매핑
        self.protein_id_to_idx = {
            protein_id: idx for idx, protein_id in enumerate(train_ids)
        }
        
        print(f"✅ ValidationEvaluator 초기화 완료")
        print(f"   단백질 개수: {len(train_ids):,}")
        print(f"   GO Terms 개수: {self.num_go_terms:,}")
        print(f"   캐시 디렉토리: {self.cache_dir}")
        print(f"   Device: {self.device}")
    
    def _compute_cache_key(self, train_terms_df: pd.DataFrame) -> str:
        """
        Label matrix 캐시 키 생성 (데이터 변경 감지용)
        
        Args:
            train_terms_df: train_terms.tsv DataFrame
        
        Returns:
            MD5 해시 문자열
        """
        # DataFrame 정렬 후 해시 생성 (재현성)
        df_sorted = train_terms_df.sort_values(['EntryID', 'GO_Term']).reset_index(drop=True)
        df_str = df_sorted.to_csv(index=False)
        hash_obj = hashlib.md5(df_str.encode())
        return hash_obj.hexdigest()[:16]
    
    def _save_label_matrix(
        self,
        label_matrix_propagated: np.ndarray,
        cache_key: str,
        metadata: Dict[str, Any]
    ):
        """
        Label matrix를 메모리 효율적으로 저장
        
        Args:
            label_matrix_propagated: 전파된 라벨 매트릭스
            cache_key: 캐시 키
            metadata: 메타데이터
        """
        cache_path = self.cache_dir / f"label_matrix_{cache_key}.npz"
        metadata_path = self.cache_dir / f"label_matrix_{cache_key}.json"
        
        # Sparse 형태로 저장 (메모리 효율)
        from scipy.sparse import csr_matrix, save_npz
        sparse_matrix = csr_matrix(label_matrix_propagated)
        save_npz(cache_path, sparse_matrix)
        
        # 메타데이터 저장
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"✅ Label matrix 저장 완료: {cache_path}")
        print(f"   Shape: {label_matrix_propagated.shape}")
        print(f"   Positive labels: {label_matrix_propagated.sum():.0f}")
        print(f"   Sparsity: {1 - sparse_matrix.nnz / np.prod(sparse_matrix.shape):.4%}")
    
    def _load_label_matrix(self, cache_key: str) -> Optional[np.ndarray]:
        """
        캐시된 label matrix 로드
        
        Args:
            cache_key: 캐시 키
        
        Returns:
            로드된 label matrix 또는 None (캐시 없음)
        """
        cache_path = self.cache_dir / f"label_matrix_{cache_key}.npz"
        metadata_path = self.cache_dir / f"label_matrix_{cache_key}.json"
        
        if not cache_path.exists() or not metadata_path.exists():
            return None
        
        try:
            from scipy.sparse import load_npz
            sparse_matrix = load_npz(cache_path)
            label_matrix = sparse_matrix.toarray().astype(np.float32)
            
            # 메타데이터 로드
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            print(f"✅ 캐시된 label matrix 로드 완료")
            print(f"   Shape: {label_matrix.shape}")
            print(f"   Positive labels: {label_matrix.sum():.0f}")
            
            return label_matrix
        
        except Exception as e:
            print(f"⚠️ 캐시 로드 실패: {e}")
            return None
    
    def build_label_matrix(self, train_terms_df: pd.DataFrame) -> np.ndarray:
        """
        Train terms에서 multi-label binary matrix 생성
        
        Args:
            train_terms_df: train_terms.tsv DataFrame
                - EntryID: 단백질 ID
                - GO_Term: GO term
                - Aspect: GO aspect (선택적)
        
        Returns:
            label_matrix: (N_proteins, N_terms) 이진 행렬
        """
        print("=" * 70)
        print("🏷️ Label Matrix 생성 중...")
        print("=" * 70)
        
        num_proteins = len(self.train_ids)
        label_matrix = np.zeros((num_proteins, self.num_go_terms), dtype=np.float32)
        
        # 라벨 할당
        matched_count = 0
        for _, row in tqdm(train_terms_df.iterrows(), total=len(train_terms_df), desc="라벨 매트릭스 구축"):
            protein_id = row['EntryID']
            go_term = row['GO_Term']
            
            if protein_id in self.protein_id_to_idx and go_term in self.term_to_idx:
                protein_idx = self.protein_id_to_idx[protein_id]
                go_idx = self.term_to_idx[go_term]
                label_matrix[protein_idx, go_idx] = 1.0
                matched_count += 1
        
        print(f"✅ Label Matrix 생성 완료")
        print(f"   Shape: {label_matrix.shape}")
        print(f"   Matched labels: {matched_count:,} / {len(train_terms_df):,}")
        print(f"   Positive labels: {label_matrix.sum():.0f}")
        
        return label_matrix
    
    @staticmethod
    def scipy_to_torch_sparse(scipy_mat):
        """scipy sparse matrix를 PyTorch sparse tensor로 변환"""
        import scipy.sparse as sp
        if not sp.issparse(scipy_mat):
            coo = sp.coo_matrix(scipy_mat)
        else:
            coo = scipy_mat.tocoo()
        values = coo.data
        indices = np.vstack((coo.row, coo.col))
        i = torch.LongTensor(indices)
        v = torch.FloatTensor(values)
        shape = coo.shape
        return torch.sparse_coo_tensor(i, v, torch.Size(shape)).coalesce()
    
    def propagate_labels_gpu(
        self,
        label_matrix: np.ndarray,
        ancestor_matrix,
        batch_size: int = 4096
    ) -> np.ndarray:
        """
        GPU 가속 라벨 전파 (계층 구조 기반)
        
        Args:
            label_matrix: (N_proteins, N_terms) 원본 라벨 행렬
            ancestor_matrix: (N_terms, N_terms) GO term ancestor 관계 행렬 (scipy sparse)
            batch_size: GPU 배치 크기
        
        Returns:
            propagated_matrix: (N_proteins, N_terms) 전파된 라벨 행렬
        """
        import scipy.sparse as sp
        
        print("\n🔄 GPU 가속 라벨 전파 중...")
        torch.cuda.empty_cache()
        
        # Ancestor matrix 준비 (전치하여 GPU로)
        if sp.issparse(ancestor_matrix):
            ancestor_T_dense = ancestor_matrix.T.toarray()
        else:
            ancestor_T_dense = ancestor_matrix.T
        
        ancestor_tensor_T = torch.tensor(ancestor_T_dense, dtype=torch.float32).to(self.device)
        num_proteins = label_matrix.shape[0]
        result_batches = []
        
        # 배치 단위로 처리
        for start_idx in tqdm(range(0, num_proteins, batch_size), desc="라벨 전파"):
            end_idx = min(start_idx + batch_size, num_proteins)
            batch_input_scipy = label_matrix[start_idx:end_idx]
            batch_input_tensor = self.scipy_to_torch_sparse(batch_input_scipy).to(self.device)
            batch_result_tensor = torch.sparse.mm(batch_input_tensor, ancestor_tensor_T)
            batch_result_tensor = (batch_result_tensor > 0).float()
            result_batches.append(batch_result_tensor.cpu().numpy())
            
            # 메모리 정리
            del batch_input_tensor, batch_result_tensor
        
        propagated_matrix = np.vstack(result_batches)
        
        print(f"✅ 라벨 전파 완료")
        print(f"   원본 positive labels: {label_matrix.sum():.0f}")
        print(f"   전파 후 positive labels: {propagated_matrix.sum():.0f}")
        print(f"   증가량: {propagated_matrix.sum() - label_matrix.sum():.0f}")
        
        # GPU 메모리 정리
        del ancestor_tensor_T
        torch.cuda.empty_cache()
        
        return propagated_matrix
    
    def prepare_label_matrix(
        self,
        train_terms_df: pd.DataFrame,
        ancestor_matrix,
        force_rebuild: bool = False
    ) -> np.ndarray:
        """
        Label matrix 생성 + 전파 (캐싱 지원)
        
        Args:
            train_terms_df: train_terms.tsv DataFrame
            ancestor_matrix: GO term ancestor 관계 행렬
            force_rebuild: 캐시 무시하고 재생성
        
        Returns:
            label_matrix_propagated: 전파된 라벨 매트릭스
        """
        # 캐시 키 생성
        cache_key = self._compute_cache_key(train_terms_df)
        
        # 캐시 확인
        if not force_rebuild:
            cached_matrix = self._load_label_matrix(cache_key)
            if cached_matrix is not None:
                return cached_matrix
        
        print("📦 캐시가 없거나 force_rebuild=True, 새로 생성합니다...")
        
        # 1. 원본 label matrix 생성
        label_matrix = self.build_label_matrix(train_terms_df)
        
        # 2. 계층 구조 전파
        label_matrix_propagated = self.propagate_labels_gpu(
            label_matrix, ancestor_matrix, batch_size=4096
        )
        
        # 3. 캐시 저장
        metadata = {
            'cache_key': cache_key,
            'shape': label_matrix_propagated.shape,
            'num_proteins': len(self.train_ids),
            'num_go_terms': self.num_go_terms,
            'positive_labels': int(label_matrix_propagated.sum()),
            'created_at': pd.Timestamp.now().isoformat()
        }
        self._save_label_matrix(label_matrix_propagated, cache_key, metadata)
        
        return label_matrix_propagated
    
    # =========================================================================
    # CAFA6 Evaluation Functions
    # =========================================================================
    
    def calculate_weighted_fmax(
        self,
        preds: np.ndarray,
        labels: np.ndarray,
        thresholds: Optional[np.ndarray] = None,
        use_gpu: bool = True,
        show_progress: bool = True
    ) -> Tuple[float, float, List[float]]:
        """
        Weighted F-max 계산 (CAFA6 공식 평가 지표)
        
        Args:
            preds: (N_proteins, N_terms) 예측 확률 (0~1)
            labels: (N_proteins, N_terms) 실제 라벨 (0 또는 1)
            thresholds: 스캔할 임계값 배열 (기본: 0~1, 101개)
            use_gpu: GPU 가속 사용 여부
            show_progress: 진행 상황 표시 여부
        
        Returns:
            best_fmax: 최고 F-max 점수
            best_threshold: 최적 임계값
            all_f1_scores: 모든 임계값에서의 F1 점수 리스트
        """
        if thresholds is None:
            thresholds = np.linspace(0, 1, 101)
        
        # GPU 가속 버전
        if use_gpu and self.device.type == 'cuda':
            with torch.no_grad():
                preds_torch = torch.tensor(preds, dtype=torch.float32, device=self.device)
                labels_torch = torch.tensor(labels, dtype=torch.float32, device=self.device)
                ic_weights_torch = torch.tensor(self.ic_weights_np, dtype=torch.float32, device=self.device)
                thresholds_torch = torch.tensor(thresholds, dtype=torch.float32, device=self.device)
                
                # Ground Truth의 총 IC
                total_true_ic = torch.matmul(labels_torch, ic_weights_torch)
                total_true_ic = torch.clamp(total_true_ic, min=1e-10)
                
                best_fmax = 0.0
                best_threshold = 0.0
                all_f1_scores = []
                
                iterator = tqdm(thresholds_torch, desc="F-max 계산", disable=not show_progress)
                for t in iterator:
                    pred_binary = (preds_torch >= t).float()
                    tp_mask = pred_binary * labels_torch
                    weighted_tp = torch.matmul(tp_mask, ic_weights_torch)
                    weighted_pred = torch.matmul(pred_binary, ic_weights_torch)
                    
                    precision = torch.where(
                        weighted_pred != 0,
                        weighted_tp / weighted_pred,
                        torch.zeros_like(weighted_tp)
                    ).mean().item()
                    
                    recall = (weighted_tp / total_true_ic).mean().item()
                    
                    if (precision + recall) > 0:
                        f1 = 2 * precision * recall / (precision + recall)
                    else:
                        f1 = 0.0
                    
                    all_f1_scores.append(f1)
                    
                    if f1 > best_fmax:
                        best_fmax = f1
                        best_threshold = t.item()
                    
                    if show_progress:
                        iterator.set_postfix({'best_fmax': f'{best_fmax:.4f}', 'threshold': f'{best_threshold:.3f}'})
                
                # GPU 메모리 정리
                del preds_torch, labels_torch, ic_weights_torch, thresholds_torch
                del total_true_ic, pred_binary, tp_mask, weighted_tp, weighted_pred
                torch.cuda.empty_cache()
                
                return best_fmax, best_threshold, all_f1_scores
        
        # CPU 버전 (fallback)
        # Ground Truth의 총 IC (Recall 분모)
        total_true_ic = labels @ self.ic_weights_np  # (N_proteins,)
        total_true_ic = np.maximum(total_true_ic, 1e-10)  # Division by zero 방지
        
        best_fmax = 0.0
        best_threshold = 0.0
        all_f1_scores = []
        
        iterator = tqdm(thresholds, desc="F-max 계산 (CPU)", disable=not show_progress)
        for t in iterator:
            # 1. 임계값 적용
            pred_binary = (preds >= t).astype(np.float32)
            
            # 2. True Positive (교집합)
            tp_mask = pred_binary * labels
            weighted_tp = tp_mask @ self.ic_weights_np  # (N_proteins,)
            
            # 3. Prediction의 총 IC (Precision 분모)
            weighted_pred = pred_binary @ self.ic_weights_np  # (N_proteins,)
            
            # 4. Weighted Precision & Recall
            precision = np.divide(
                weighted_tp, weighted_pred,
                out=np.zeros_like(weighted_tp),
                where=weighted_pred != 0
            )
            recall = weighted_tp / total_true_ic
            
            # 5. 평균
            avg_prec = np.mean(precision)
            avg_rec = np.mean(recall)
            
            # 6. F1 계산
            if (avg_prec + avg_rec) > 0:
                f1 = 2 * avg_prec * avg_rec / (avg_prec + avg_rec)
            else:
                f1 = 0.0
            
            all_f1_scores.append(f1)
            
            # 7. 최고 F-max 업데이트
            if f1 > best_fmax:
                best_fmax = f1
                best_threshold = t
            
            if show_progress:
                iterator.set_postfix({'best_fmax': f'{best_fmax:.4f}', 'threshold': f'{best_threshold:.3f}'})
        
        return best_fmax, best_threshold, all_f1_scores
    
    def calculate_smin(
        self,
        preds: np.ndarray,
        labels: np.ndarray,
        thresholds: Optional[np.ndarray] = None,
        use_gpu: bool = True,
        show_progress: bool = True
    ) -> Tuple[float, float]:
        """
        S-min (Semantic Distance) 계산
        
        Args:
            preds: (N_proteins, N_terms) 예측 확률 (0~1)
            labels: (N_proteins, N_terms) 실제 라벨 (0 또는 1)
            thresholds: 스캔할 임계값 배열
            use_gpu: GPU 가속 사용 여부
            show_progress: 진행 상황 표시 여부
        
        Returns:
            min_s: 최소 S-score
            best_threshold: 최적 임계값
        """
        if thresholds is None:
            thresholds = np.linspace(0, 1, 101)
        
        # GPU 가속 버전
        if use_gpu and self.device.type == 'cuda':
            with torch.no_grad():
                preds_torch = torch.tensor(preds, dtype=torch.float32, device=self.device)
                labels_torch = torch.tensor(labels, dtype=torch.float32, device=self.device)
                ic_weights_torch = torch.tensor(self.ic_weights_np, dtype=torch.float32, device=self.device)
                thresholds_torch = torch.tensor(thresholds, dtype=torch.float32, device=self.device)
                
                min_s = float('inf')
                best_threshold = 0.0
                
                iterator = tqdm(thresholds_torch, desc="S-min 계산", disable=not show_progress)
                for t in iterator:
                    pred_binary = (preds_torch >= t).float()
                    
                    # False Negatives (RU)
                    fn_mask = labels_torch * (1 - pred_binary)
                    avg_ru = torch.matmul(fn_mask, ic_weights_torch).mean().item()
                    
                    # False Positives (MI)
                    fp_mask = pred_binary * (1 - labels_torch)
                    avg_mi = torch.matmul(fp_mask, ic_weights_torch).mean().item()
                    
                    # Euclidean Distance
                    s_score = np.sqrt(avg_ru**2 + avg_mi**2)
                    
                    if s_score < min_s:
                        min_s = s_score
                        best_threshold = t.item()
                    
                    if show_progress:
                        iterator.set_postfix({'min_s': f'{min_s:.4f}', 'threshold': f'{best_threshold:.3f}'})
                
                # GPU 메모리 정리
                del preds_torch, labels_torch, ic_weights_torch, thresholds_torch
                del pred_binary, fn_mask, fp_mask
                torch.cuda.empty_cache()
                
                return min_s, best_threshold
        
        # CPU 버전 (fallback)
        min_s = float('inf')
        best_threshold = 0.0
        
        iterator = tqdm(thresholds, desc="S-min 계산 (CPU)", disable=not show_progress)
        for t in iterator:
            pred_binary = (preds >= t).astype(np.float32)
            
            # False Negatives (RU: Remaining Uncertainty)
            fn_mask = labels * (1 - pred_binary)
            ru_per_protein = fn_mask @ self.ic_weights_np
            avg_ru = np.mean(ru_per_protein)
            
            # False Positives (MI: Misinformation)
            fp_mask = pred_binary * (1 - labels)
            mi_per_protein = fp_mask @ self.ic_weights_np
            avg_mi = np.mean(mi_per_protein)
            
            # Euclidean Distance
            s_score = np.sqrt(avg_ru**2 + avg_mi**2)
            
            if s_score < min_s:
                min_s = s_score
                best_threshold = t
            
            if show_progress:
                iterator.set_postfix({'min_s': f'{min_s:.4f}', 'threshold': f'{best_threshold:.3f}'})
        
        return min_s, best_threshold
    
    def evaluate_cafa6(
        self,
        predictions: np.ndarray,
        labels: np.ndarray,
        thresholds: Optional[np.ndarray] = None,
        use_gpu: bool = True,
        show_progress: bool = True,
        verbose: bool = True
    ) -> Dict[str, Any]:
        """
        CAFA6 전체 평가 지표 계산
        
        Args:
            predictions: (N_proteins, N_terms) 예측 확률
            labels: (N_proteins, N_terms) 실제 라벨
            thresholds: 임계값 배열 (기본: 0~1, 101개)
            use_gpu: GPU 가속 사용 여부
            show_progress: 진행 상황 표시 여부
            verbose: 결과 출력 여부
        
        Returns:
            dict: {
                'fmax': float,
                'fmax_threshold': float,
                'smin': float,
                'smin_threshold': float,
                'f1_scores': list
            }
        """
        if thresholds is None:
            thresholds = np.linspace(0, 1, 101)
        
        if verbose:
            print("\n" + "=" * 70)
            print("⭐ CAFA6 공식 평가 지표 계산")
            print("=" * 70)
            if use_gpu and self.device.type == 'cuda':
                print(f"   ⚡ GPU 가속: ✅ Enabled ({self.device})")
            else:
                print(f"   ⚡ GPU 가속: ❌ CPU 모드")
            print(f"   📊 임계값 개수: {len(thresholds)}")
        
        # Weighted F-max
        fmax, fmax_t, f1_scores = self.calculate_weighted_fmax(
            predictions, labels, thresholds, use_gpu=use_gpu, show_progress=show_progress
        )
        
        # S-min
        smin, smin_t = self.calculate_smin(
            predictions, labels, thresholds, use_gpu=use_gpu, show_progress=show_progress
        )
        
        results = {
            'fmax': fmax,
            'fmax_threshold': fmax_t,
            'smin': smin,
            'smin_threshold': smin_t,
            'f1_scores': f1_scores
        }
        
        if verbose:
            print("=" * 70)
            print("📊 CAFA6 평가 결과")
            print("=" * 70)
            print(f"  Weighted F-max: {fmax:.6f} (threshold={fmax_t:.4f})")
            print(f"  S-min: {smin:.6f} (threshold={smin_t:.4f})")
            print("=" * 70)
        
        return results
    
    # =========================================================================
    # Sklearn Metrics at Various Thresholds
    # =========================================================================
    
    def evaluate_at_threshold(
        self,
        predictions: np.ndarray,
        labels: np.ndarray,
        threshold: float,
        use_gpu: bool = False
    ) -> Dict[str, float]:
        """
        특정 임계값에서 sklearn 메트릭 계산
        
        Args:
            predictions: (N_proteins, N_terms) 예측 확률
            labels: (N_proteins, N_terms) 실제 라벨
            threshold: 임계값
            use_gpu: GPU 가속 사용 여부 (이진화에만 적용)
        
        Returns:
            메트릭 딕셔너리
        """
        # GPU 가속 이진화
        if use_gpu and self.device.type == 'cuda':
            with torch.no_grad():
                preds_torch = torch.tensor(predictions, dtype=torch.float32, device=self.device)
                y_pred_binary = (preds_torch >= threshold).cpu().numpy().astype(int)
                del preds_torch
        else:
            # CPU 이진화
            y_pred_binary = (predictions >= threshold).astype(int)
        
        # Sklearn 메트릭 (CPU에서 계산)
        f1_micro = f1_score(labels, y_pred_binary, average='micro', zero_division=0)
        f1_macro = f1_score(labels, y_pred_binary, average='macro', zero_division=0)
        precision = precision_score(labels, y_pred_binary, average='micro', zero_division=0)
        recall = recall_score(labels, y_pred_binary, average='micro', zero_division=0)
        hamming = hamming_loss(labels, y_pred_binary)
        
        # 예측 통계
        avg_pred_terms = y_pred_binary.sum(axis=1).mean()
        
        return {
            'threshold': threshold,
            'f1_micro': f1_micro,
            'f1_macro': f1_macro,
            'precision': precision,
            'recall': recall,
            'hamming_loss': hamming,
            'avg_pred_terms': avg_pred_terms
        }
    
    def evaluate_thresholds(
        self,
        predictions: np.ndarray,
        labels: np.ndarray,
        thresholds: List[float],
        use_gpu: bool = True,
        show_progress: bool = True,
        verbose: bool = True
    ) -> List[Dict[str, float]]:
        """
        여러 임계값에서 sklearn 메트릭 평가
        
        Args:
            predictions: (N_proteins, N_terms) 예측 확률
            labels: (N_proteins, N_terms) 실제 라벨
            thresholds: 평가할 임계값 리스트
            use_gpu: GPU 가속 사용 여부
            show_progress: 진행 상황 표시 여부
            verbose: 결과 출력 여부
        
        Returns:
            각 임계값에서의 메트릭 리스트
        """
        if verbose:
            print("\n" + "=" * 70)
            print("📊 다양한 임계값에서 Sklearn 메트릭 평가")
            print("=" * 70)
            if use_gpu and self.device.type == 'cuda':
                print(f"   ⚡ GPU 가속: ✅ Enabled ({self.device})")
            else:
                print(f"   ⚡ GPU 가속: ❌ CPU 모드")
            print(f"   📊 임계값 개수: {len(thresholds)}")
        
        results = []
        iterator = tqdm(thresholds, desc="임계값 평가", disable=not show_progress)
        for t in iterator:
            result = self.evaluate_at_threshold(predictions, labels, t, use_gpu=use_gpu)
            results.append(result)
            
            if show_progress:
                iterator.set_postfix({
                    'F1-Micro': f"{result['f1_micro']:.4f}",
                    'threshold': f"{t:.3f}"
                })
        
        if verbose:
            # 결과 테이블 출력
            print(f"\n{'Threshold':<12} {'F1-Micro':<12} {'F1-Macro':<12} {'Precision':<12} {'Recall':<12} {'Hamming':<12} {'Avg Terms':<12}")
            print("-" * 84)
            for res in results:
                print(f"{res['threshold']:<12.4f} {res['f1_micro']:<12.4f} {res['f1_macro']:<12.4f} "
                      f"{res['precision']:<12.4f} {res['recall']:<12.4f} {res['hamming_loss']:<12.6f} "
                      f"{res['avg_pred_terms']:<12.2f}")
            print("=" * 84)
        
        return results
    
    # =========================================================================
    # Combined Evaluation (CAFA6 + Sklearn)
    # =========================================================================
    
    def full_evaluation(
        self,
        predictions: np.ndarray,
        labels: np.ndarray,
        cafa6_thresholds: Optional[np.ndarray] = None,
        sklearn_thresholds: Optional[List[float]] = None,
        verbose: bool = True
    ) -> Dict[str, Any]:
        """
        전체 평가 (CAFA6 + Sklearn)
        
        Args:
            predictions: 예측 확률
            labels: 실제 라벨
            cafa6_thresholds: CAFA6 평가용 임계값 배열 (기본: 0~1, 101개)
            sklearn_thresholds: Sklearn 평가용 임계값 리스트 (기본: CAFA6 최적값)
            verbose: 결과 출력 여부
        
        Returns:
            전체 평가 결과 딕셔너리
        """
        # CAFA6 평가
        cafa6_results = self.evaluate_cafa6(
            predictions, labels, cafa6_thresholds, verbose
        )
        
        # Sklearn 평가 (CAFA6 최적 임계값 사용)
        if sklearn_thresholds is None:
            sklearn_thresholds = [cafa6_results['fmax_threshold']]
        
        sklearn_results = self.evaluate_thresholds(
            predictions, labels, sklearn_thresholds, verbose
        )
        
        return {
            'cafa6': cafa6_results,
            'sklearn': sklearn_results
        }
    
    # =========================================================================
    # Ancestral Resonance Post-Processing
    # =========================================================================
    
    def apply_ancestral_resonance(
        self,
        predictions: np.ndarray,
        ancestor_matrix_sparse: Union[csr_matrix, torch.Tensor],
        alpha: float = 0.5,
        use_gpu: bool = True,
        show_progress: bool = True
    ) -> np.ndarray:
        """
        Ancestral Resonance를 사용한 예측 스무딩 (계층 구조 기반 후처리)
        
        원리:
            Final_Score = (Self_Score ^ alpha) * (Ancestral_Support ^ (1-alpha))
            Geometric Mean을 사용하여 soft AND gate 역할
        
        Args:
            predictions: (N_proteins, N_terms) 예측 확률
            ancestor_matrix_sparse: (N_terms, N_terms) 조상 행렬
                - A[i, j] = 1이면 j는 i의 조상
            alpha: self-score 가중치 (0.0 ~ 1.0)
                - 0.5: Geometric Mean (균형)
                - 0.8: 자신의 점수를 더 신뢰
                - 0.2: 조상 점수를 더 신뢰
            use_gpu: GPU 가속 사용 여부
            show_progress: 진행 상황 표시 여부
        
        Returns:
            (N_proteins, N_terms) Resonated 예측 확률
        """
        if use_gpu and self.device.type == 'cuda':
            with torch.no_grad():
                if show_progress:
                    print("\n" + "=" * 70)
                    print("🔬 Ancestral Resonance 적용 (GPU)")
                    print("=" * 70)
                    print(f"   Alpha: {alpha:.2f}")
                    print(f"   Device: {self.device}")
                
                # 1. 입력을 GPU 텐서로 변환
                if show_progress:
                    print("   [1/5] GPU 텐서 변환 중...")
                
                preds_tensor = torch.tensor(predictions, device=self.device, dtype=torch.float32)
                
                # 2. Sparse matrix를 torch sparse tensor로 변환
                if not isinstance(ancestor_matrix_sparse, torch.Tensor):
                    if show_progress:
                        print("   [2/5] Ancestor matrix 변환 중...")
                    
                    coo = ancestor_matrix_sparse.tocoo()
                    indices = torch.LongTensor(np.vstack((coo.row, coo.col))).to(self.device)
                    values = torch.FloatTensor(coo.data).to(self.device)
                    shape = coo.shape
                    adj = torch.sparse_coo_tensor(indices, values, torch.Size(shape), device=self.device)
                else:
                    adj = ancestor_matrix_sparse
                
                # 3. Ancestral Support 계산
                if show_progress:
                    print("   [3/5] Ancestral Support 계산 중...")
                
                # Ancestor Sum: (L, L) @ (L, N) -> (L, N) -> transpose -> (N, L)
                ancestor_sum = torch.sparse.mm(adj, preds_tensor.T).T
                
                # Ancestor Count: 각 term이 가진 조상의 수
                ones_vec = torch.ones(adj.shape[1], 1, device=self.device)
                ancestor_counts = torch.sparse.mm(adj, ones_vec).T  # (1, L)
                ancestor_counts = torch.clamp(ancestor_counts, min=1.0)
                
                # Ancestral Support (평균)
                ancestor_support = ancestor_sum / ancestor_counts
                
                # 4. Resonance Fusion (Geometric Mean)
                if show_progress:
                    print("   [4/5] Geometric Mean 적용 중...")
                
                resonated_preds = torch.pow(preds_tensor, alpha) * torch.pow(ancestor_support, (1.0 - alpha))
                
                # 5. CPU로 변환
                if show_progress:
                    print("   [5/5] 결과 변환 중...")
                
                result = resonated_preds.cpu().numpy()
                
                # GPU 메모리 정리
                del preds_tensor, adj, ancestor_sum, ancestor_counts, ancestor_support, resonated_preds
                torch.cuda.empty_cache()
                
                if show_progress:
                    print("   ✅ Ancestral Resonance 완료")
                    print("=" * 70)
                
                return result
        
        else:
            # CPU 버전
            if show_progress:
                print("\n" + "=" * 70)
                print("🔬 Ancestral Resonance 적용 (CPU)")
                print("=" * 70)
            
            # Scipy sparse matrix 사용
            if show_progress:
                print("   [1/3] Ancestral Support 계산 중...")
            
            # (N, L) @ (L, L).T = (N, L)
            ancestor_sum = predictions @ ancestor_matrix_sparse.T
            
            # 각 term의 조상 개수
            ancestor_counts = np.array(ancestor_matrix_sparse.sum(axis=1)).flatten()
            ancestor_counts = np.maximum(ancestor_counts, 1.0)
            
            # Ancestral Support
            ancestor_support = ancestor_sum / ancestor_counts
            
            # Geometric Mean
            if show_progress:
                print("   [2/3] Geometric Mean 적용 중...")
            
            resonated_preds = np.power(predictions, alpha) * np.power(ancestor_support, (1.0 - alpha))
            
            if show_progress:
                print("   [3/3] 완료")
                print("   ✅ Ancestral Resonance 완료")
                print("=" * 70)
            
            return resonated_preds
    
    def evaluate_with_ancestral_resonance(
        self,
        predictions: np.ndarray,
        labels: np.ndarray,
        ancestor_matrix_sparse: Union[csr_matrix, torch.Tensor],
        alpha: float = 0.5,
        base_threshold: Optional[float] = None,
        ic_pruning_alpha: float = 0.03,
        use_gpu: bool = True,
        show_progress: bool = True,
        verbose: bool = True
    ) -> Dict[str, Any]:
        """
        Ancestral Resonance 후처리를 적용하여 CAFA6 지표 재평가
        
        Args:
            predictions: (N_proteins, N_terms) 원본 예측 확률
            labels: (N_proteins, N_terms) 실제 라벨
            ancestor_matrix_sparse: (N_terms, N_terms) 조상 행렬
            alpha: Ancestral Resonance alpha (기본: 0.5)
            base_threshold: IC-adaptive pruning의 기본 임계값 (None이면 사용 안함)
            ic_pruning_alpha: IC-adaptive pruning의 alpha 값
            use_gpu: GPU 가속 사용 여부
            show_progress: 진행 상황 표시 여부
            verbose: 결과 출력 여부
        
        Returns:
            dict: {
                'original': {원본 CAFA6 결과},
                'resonated': {Resonance 적용 후 CAFA6 결과},
                'pruned': {IC-pruning 적용 후 CAFA6 결과} (optional)
            }
        """
        if verbose:
            print("\n" + "=" * 70)
            print("⭐ Ancestral Resonance 후처리 평가")
            print("=" * 70)
        
        results = {}
        
        # 1. 원본 평가
        if verbose:
            print("\n[1단계] 원본 예측 평가")
        
        results['original'] = self.evaluate_cafa6(
            predictions=predictions,
            labels=labels,
            use_gpu=use_gpu,
            show_progress=show_progress,
            verbose=verbose
        )
        
        # 2. Ancestral Resonance 적용
        resonated_preds = self.apply_ancestral_resonance(
            predictions=predictions,
            ancestor_matrix_sparse=ancestor_matrix_sparse,
            alpha=alpha,
            use_gpu=use_gpu,
            show_progress=show_progress
        )
        
        # 3. Resonated 예측 평가
        if verbose:
            print("\n[2단계] Resonated 예측 평가")
        
        results['resonated'] = self.evaluate_cafa6(
            predictions=resonated_preds,
            labels=labels,
            use_gpu=use_gpu,
            show_progress=show_progress,
            verbose=verbose
        )
        
        # 4. IC-Adaptive Pruning (선택적)
        if base_threshold is not None:
            if verbose:
                print("\n[3단계] IC-Adaptive Pruning 적용")
                print("=" * 70)
                print(f"   Base Threshold: {base_threshold:.4f}")
                print(f"   IC Pruning Alpha: {ic_pruning_alpha:.4f}")
            
            if use_gpu and self.device.type == 'cuda':
                with torch.no_grad():
                    resonated_tensor = torch.tensor(resonated_preds, device=self.device, dtype=torch.float32)
                    ic_weights_tensor = torch.tensor(self.ic_weights_np, device=self.device, dtype=torch.float32).unsqueeze(0)
                    
                    dynamic_thr = base_threshold + (ic_pruning_alpha * ic_weights_tensor)
                    mask = resonated_tensor >= dynamic_thr
                    pruned_preds = (resonated_tensor * mask.float()).cpu().numpy()
                    
                    del resonated_tensor, ic_weights_tensor, dynamic_thr, mask
                    torch.cuda.empty_cache()
            else:
                dynamic_thr = base_threshold + (ic_pruning_alpha * self.ic_weights_np)
                mask = resonated_preds >= dynamic_thr
                pruned_preds = resonated_preds * mask
            
            if verbose:
                print("   ✅ Pruning 완료")
                print("\n[4단계] Pruned 예측 평가")
            
            results['pruned'] = self.evaluate_cafa6(
                predictions=pruned_preds,
                labels=labels,
                use_gpu=use_gpu,
                show_progress=show_progress,
                verbose=verbose
            )
        
        # 5. 비교 결과 출력
        if verbose:
            print("\n" + "=" * 70)
            print("📊 후처리 효과 비교")
            print("=" * 70)
            
            stages = ['original', 'resonated']
            if 'pruned' in results:
                stages.append('pruned')
            
            print(f"\n{'Stage':<15} {'F-max':<12} {'F-max Δ':<12} {'S-min':<12} {'S-min Δ':<12}")
            print("-" * 63)
            
            baseline_fmax = results['original']['fmax']
            baseline_smin = results['original']['smin']
            
            for stage in stages:
                fmax = results[stage]['fmax']
                smin = results[stage]['smin']
                fmax_delta = fmax - baseline_fmax
                smin_delta = smin - baseline_smin
                
                print(f"{stage:<15} {fmax:<12.6f} {fmax_delta:>+11.6f} {smin:<12.6f} {smin_delta:>+11.6f}")
            
            print("=" * 70)
        
        return results
    
    # =========================================================================
    # Ancestral Resonance Post-Processing
    # =========================================================================
    
    def apply_ancestral_resonance(
        self,
        predictions: np.ndarray,
        ancestor_matrix_sparse: Union[csr_matrix, torch.Tensor],
        alpha: float = 0.5,
        use_gpu: bool = True,
        show_progress: bool = True
    ) -> np.ndarray:
        """
        Ancestral Resonance를 사용한 예측 스무딩 (계층 구조 기반 후처리)
        
        원리:
            Final_Score = (Self_Score ^ alpha) * (Ancestral_Support ^ (1-alpha))
            Geometric Mean을 사용하여 soft AND gate 역할
        
        Args:
            predictions: (N_proteins, N_terms) 예측 확률
            ancestor_matrix_sparse: (N_terms, N_terms) 조상 행렬
                - A[i, j] = 1이면 j는 i의 조상
            alpha: self-score 가중치 (0.0 ~ 1.0)
                - 0.5: Geometric Mean (균형)
                - 0.8: 자신의 점수를 더 신뢰
                - 0.2: 조상 점수를 더 신뢰
            use_gpu: GPU 가속 사용 여부
            show_progress: 진행 상황 표시 여부
        
        Returns:
            (N_proteins, N_terms) Resonated 예측 확률
        """
        if use_gpu and self.device.type == 'cuda':
            with torch.no_grad():
                if show_progress:
                    print("\n" + "=" * 70)
                    print("🔬 Ancestral Resonance 적용 (GPU)")
                    print("=" * 70)
                    print(f"   Alpha: {alpha:.2f}")
                    print(f"   Device: {self.device}")
                
                # 1. 입력을 GPU 텐서로 변환
                if show_progress:
                    print("   [1/5] GPU 텐서 변환 중...")
                
                preds_tensor = torch.tensor(predictions, device=self.device, dtype=torch.float32)
                
                # 2. Sparse matrix를 torch sparse tensor로 변환
                if not isinstance(ancestor_matrix_sparse, torch.Tensor):
                    if show_progress:
                        print("   [2/5] Ancestor matrix 변환 중...")
                    
                    coo = ancestor_matrix_sparse.tocoo()
                    indices = torch.LongTensor(np.vstack((coo.row, coo.col))).to(self.device)
                    values = torch.FloatTensor(coo.data).to(self.device)
                    shape = coo.shape
                    adj = torch.sparse_coo_tensor(indices, values, torch.Size(shape), device=self.device)
                else:
                    adj = ancestor_matrix_sparse
                
                # 3. Ancestral Support 계산
                if show_progress:
                    print("   [3/5] Ancestral Support 계산 중...")
                
                # Ancestor Sum: (L, L) @ (L, N) -> (L, N) -> transpose -> (N, L)
                ancestor_sum = torch.sparse.mm(adj, preds_tensor.T).T
                
                # Ancestor Count: 각 term이 가진 조상의 수
                ones_vec = torch.ones(adj.shape[1], 1, device=self.device)
                ancestor_counts = torch.sparse.mm(adj, ones_vec).T  # (1, L)
                ancestor_counts = torch.clamp(ancestor_counts, min=1.0)
                
                # Ancestral Support (평균)
                ancestor_support = ancestor_sum / ancestor_counts
                
                # 4. Resonance Fusion (Geometric Mean)
                if show_progress:
                    print("   [4/5] Geometric Mean 적용 중...")
                
                resonated_preds = torch.pow(preds_tensor, alpha) * torch.pow(ancestor_support, (1.0 - alpha))
                
                # 5. CPU로 변환
                if show_progress:
                    print("   [5/5] 결과 변환 중...")
                
                result = resonated_preds.cpu().numpy()
                
                # GPU 메모리 정리
                del preds_tensor, adj, ancestor_sum, ancestor_counts, ancestor_support, resonated_preds
                torch.cuda.empty_cache()
                
                if show_progress:
                    print("   ✅ Ancestral Resonance 완료")
                    print("=" * 70)
                
                return result
        
        else:
            # CPU 버전
            if show_progress:
                print("\n" + "=" * 70)
                print("🔬 Ancestral Resonance 적용 (CPU)")
                print("=" * 70)
            
            # Scipy sparse matrix 사용
            if show_progress:
                print("   [1/3] Ancestral Support 계산 중...")
            
            # (N, L) @ (L, L).T = (N, L)
            ancestor_sum = predictions @ ancestor_matrix_sparse.T
            
            # 각 term의 조상 개수
            ancestor_counts = np.array(ancestor_matrix_sparse.sum(axis=1)).flatten()
            ancestor_counts = np.maximum(ancestor_counts, 1.0)
            
            # Ancestral Support
            ancestor_support = ancestor_sum / ancestor_counts
            
            # Geometric Mean
            if show_progress:
                print("   [2/3] Geometric Mean 적용 중...")
            
            resonated_preds = np.power(predictions, alpha) * np.power(ancestor_support, (1.0 - alpha))
            
            if show_progress:
                print("   [3/3] 완료")
                print("   ✅ Ancestral Resonance 완료")
                print("=" * 70)
            
            return resonated_preds
    
    def evaluate_with_ancestral_resonance(
        self,
        predictions: np.ndarray,
        labels: np.ndarray,
        ancestor_matrix_sparse: Union[csr_matrix, torch.Tensor],
        alpha: float = 0.5,
        base_threshold: Optional[float] = None,
        ic_pruning_alpha: float = 0.03,
        use_gpu: bool = True,
        show_progress: bool = True,
        verbose: bool = True
    ) -> Dict[str, Any]:
        """
        Ancestral Resonance 후처리를 적용하여 CAFA6 지표 재평가
        
        Args:
            predictions: (N_proteins, N_terms) 원본 예측 확률
            labels: (N_proteins, N_terms) 실제 라벨
            ancestor_matrix_sparse: (N_terms, N_terms) 조상 행렬
            alpha: Ancestral Resonance alpha (기본: 0.5)
            base_threshold: IC-adaptive pruning의 기본 임계값 (None이면 사용 안함)
            ic_pruning_alpha: IC-adaptive pruning의 alpha 값
            use_gpu: GPU 가속 사용 여부
            show_progress: 진행 상황 표시 여부
            verbose: 결과 출력 여부
        
        Returns:
            dict: {
                'original': {원본 CAFA6 결과},
                'resonated': {Resonance 적용 후 CAFA6 결과},
                'pruned': {IC-pruning 적용 후 CAFA6 결과} (optional)
            }
        """
        if verbose:
            print("\n" + "=" * 70)
            print("⭐ Ancestral Resonance 후처리 평가")
            print("=" * 70)
        
        results = {}
        
        # 1. 원본 평가
        if verbose:
            print("\n[1단계] 원본 예측 평가")
        
        results['original'] = self.evaluate_cafa6(
            predictions=predictions,
            labels=labels,
            use_gpu=use_gpu,
            show_progress=show_progress,
            verbose=verbose
        )
        
        # 2. Ancestral Resonance 적용
        resonated_preds = self.apply_ancestral_resonance(
            predictions=predictions,
            ancestor_matrix_sparse=ancestor_matrix_sparse,
            alpha=alpha,
            use_gpu=use_gpu,
            show_progress=show_progress
        )
        
        # 3. Resonated 예측 평가
        if verbose:
            print("\n[2단계] Resonated 예측 평가")
        
        results['resonated'] = self.evaluate_cafa6(
            predictions=resonated_preds,
            labels=labels,
            use_gpu=use_gpu,
            show_progress=show_progress,
            verbose=verbose
        )
        
        # 4. IC-Adaptive Pruning (선택적)
        if base_threshold is not None:
            if verbose:
                print("\n[3단계] IC-Adaptive Pruning 적용")
                print("=" * 70)
                print(f"   Base Threshold: {base_threshold:.4f}")
                print(f"   IC Pruning Alpha: {ic_pruning_alpha:.4f}")
            
            if use_gpu and self.device.type == 'cuda':
                with torch.no_grad():
                    resonated_tensor = torch.tensor(resonated_preds, device=self.device, dtype=torch.float32)
                    ic_weights_tensor = torch.tensor(self.ic_weights_np, device=self.device, dtype=torch.float32).unsqueeze(0)
                    
                    dynamic_thr = base_threshold + (ic_pruning_alpha * ic_weights_tensor)
                    mask = resonated_tensor >= dynamic_thr
                    pruned_preds = (resonated_tensor * mask.float()).cpu().numpy()
                    
                    del resonated_tensor, ic_weights_tensor, dynamic_thr, mask
                    torch.cuda.empty_cache()
            else:
                dynamic_thr = base_threshold + (ic_pruning_alpha * self.ic_weights_np)
                mask = resonated_preds >= dynamic_thr
                pruned_preds = resonated_preds * mask
            
            if verbose:
                print("   ✅ Pruning 완료")
                print("\n[4단계] Pruned 예측 평가")
            
            results['pruned'] = self.evaluate_cafa6(
                predictions=pruned_preds,
                labels=labels,
                use_gpu=use_gpu,
                show_progress=show_progress,
                verbose=verbose
            )
        
        # 5. 비교 결과 출력
        if verbose:
            print("\n" + "=" * 70)
            print("📊 후처리 효과 비교")
            print("=" * 70)
            
            stages = ['original', 'resonated']
            if 'pruned' in results:
                stages.append('pruned')
            
            print(f"\n{'Stage':<15} {'F-max':<12} {'F-max Δ':<12} {'S-min':<12} {'S-min Δ':<12}")
            print("-" * 63)
            
            baseline_fmax = results['original']['fmax']
            baseline_smin = results['original']['smin']
            
            for stage in stages:
                fmax = results[stage]['fmax']
                smin = results[stage]['smin']
                fmax_delta = fmax - baseline_fmax
                smin_delta = smin - baseline_smin
                
                print(f"{stage:<15} {fmax:<12.6f} {fmax_delta:>+11.6f} {smin:<12.6f} {smin_delta:>+11.6f}")
            
            print("=" * 70)
        
        return results
    
    # =========================================================================
    # IC-Adaptive Dynamic Thresholding
    # =========================================================================
    
    def dynamic_thresholding(
        self,
        predictions: np.ndarray,
        labels: np.ndarray,
        base_threshold: float,
        alpha_values: List[float] = [0.02, 0.03, 0.04, 0.05],
        use_gpu: bool = True,
        show_progress: bool = True,
        verbose: bool = True
    ) -> List[Dict[str, Any]]:
        """
        IC-Adaptive Dynamic Thresholding 평가
        
        Args:
            predictions: (N_proteins, N_terms) 예측 확률
            labels: (N_proteins, N_terms) 실제 라벨
            base_threshold: 기본 임계값
            alpha_values: 테스트할 alpha 값 리스트
            use_gpu: GPU 가속 사용 여부
            show_progress: 진행 상황 표시 여부
            verbose: 결과 출력 여부
        
        Returns:
            각 alpha 값에서의 CAFA6 메트릭 리스트
        """
        if verbose:
            print("\n" + "=" * 70)
            print("🔧 IC-Adaptive Dynamic Thresholding")
            print("=" * 70)
            print(f"   Base Threshold: {base_threshold:.4f}")
            print(f"   Alpha values: {alpha_values}")
            print(f"   GPU acceleration: {'✅ Enabled' if use_gpu and self.device.type == 'cuda' else '❌ Disabled'}")
        
        # IC 가중치 정규화
        ic_weights_normalized = (
            (self.ic_weights_np - self.ic_weights_np.min()) /
            (self.ic_weights_np.max() - self.ic_weights_np.min() + 1e-10)
        )
        
        results = []
        
        if use_gpu and self.device.type == 'cuda':
            # GPU 가속 버전
            with torch.no_grad():
                preds_torch = torch.tensor(predictions, dtype=torch.float32, device=self.device)
                labels_torch = torch.tensor(labels, dtype=torch.float32, device=self.device)
                ic_weights_torch = torch.tensor(self.ic_weights_np, dtype=torch.float32, device=self.device)
                ic_norm_torch = torch.tensor(ic_weights_normalized, dtype=torch.float32, device=self.device)
                
                # Ground Truth IC
                total_true_ic = torch.matmul(labels_torch, ic_weights_torch)
                total_true_ic = torch.clamp(total_true_ic, min=1e-10)
                
                iterator = tqdm(alpha_values, desc="Dynamic Thresholding (GPU)", disable=not show_progress)
                for alpha in iterator:
                    # Dynamic thresholds
                    dynamic_thresholds = base_threshold + (alpha * ic_norm_torch)
                    dynamic_thresholds = torch.clamp(dynamic_thresholds, 0, 1)
                    
                    # 예측 이진화
                    y_pred_binary = (preds_torch >= dynamic_thresholds.unsqueeze(0)).float()
                    
                    # Weighted F-max
                    tp_mask = y_pred_binary * labels_torch
                    weighted_tp = torch.matmul(tp_mask, ic_weights_torch)
                    weighted_pred = torch.matmul(y_pred_binary, ic_weights_torch)
                    
                    precision_weighted = torch.where(
                        weighted_pred != 0,
                        weighted_tp / weighted_pred,
                        torch.zeros_like(weighted_tp)
                    ).mean().item()
                    
                    recall_weighted = (weighted_tp / total_true_ic).mean().item()
                    
                    if (precision_weighted + recall_weighted) > 0:
                        weighted_fmax = 2 * precision_weighted * recall_weighted / (
                            precision_weighted + recall_weighted
                        )
                    else:
                        weighted_fmax = 0.0
                    
                    # S-min
                    fn_mask = labels_torch * (1 - y_pred_binary)
                    ru = torch.matmul(fn_mask, ic_weights_torch).mean().item()
                    
                    fp_mask = y_pred_binary * (1 - labels_torch)
                    mi = torch.matmul(fp_mask, ic_weights_torch).mean().item()
                    
                    smin = np.sqrt(ru**2 + mi**2)
                    
                    results.append({
                        'alpha': alpha,
                        'weighted_fmax': weighted_fmax,
                        'smin': smin,
                        'avg_threshold': dynamic_thresholds.mean().item()
                    })
                    
                    if show_progress:
                        iterator.set_postfix({
                            'F-max': f'{weighted_fmax:.4f}',
                            'S-min': f'{smin:.4f}'
                        })
                
                # GPU 메모리 정리
                del preds_torch, labels_torch, ic_weights_torch, ic_norm_torch
                del total_true_ic, dynamic_thresholds, y_pred_binary
                torch.cuda.empty_cache()
        
        else:
            # CPU 버전
            total_true_ic = labels @ self.ic_weights_np
            total_true_ic = np.maximum(total_true_ic, 1e-10)
            
            iterator = tqdm(alpha_values, desc="Dynamic Thresholding (CPU)", disable=not show_progress)
            for alpha in iterator:
                # Dynamic thresholds
                dynamic_thresholds = base_threshold + (alpha * ic_weights_normalized)
                dynamic_thresholds = np.clip(dynamic_thresholds, 0, 1)
                
                # 예측 이진화
                y_pred_binary = (predictions >= dynamic_thresholds).astype(np.float32)
                
                # Weighted F-max
                tp_mask = y_pred_binary * labels
                weighted_tp = tp_mask @ self.ic_weights_np
                weighted_pred = y_pred_binary @ self.ic_weights_np
                
                precision_weighted = np.divide(
                    weighted_tp, weighted_pred,
                    out=np.zeros_like(weighted_tp),
                    where=weighted_pred != 0
                ).mean()
                
                recall_weighted = (weighted_tp / total_true_ic).mean()
                
                if (precision_weighted + recall_weighted) > 0:
                    weighted_fmax = 2 * precision_weighted * recall_weighted / (
                        precision_weighted + recall_weighted
                    )
                else:
                    weighted_fmax = 0.0
                
                # S-min
                fn_mask = labels * (1 - y_pred_binary)
                ru = (fn_mask @ self.ic_weights_np).mean()
                
                fp_mask = y_pred_binary * (1 - labels)
                mi = (fp_mask @ self.ic_weights_np).mean()
                
                smin = np.sqrt(ru**2 + mi**2)
                
                results.append({
                    'alpha': alpha,
                    'weighted_fmax': weighted_fmax,
                    'smin': smin,
                    'avg_threshold': dynamic_thresholds.mean()
                })
                
                if show_progress:
                    iterator.set_postfix({
                        'F-max': f'{weighted_fmax:.4f}',
                        'S-min': f'{smin:.4f}'
                    })
        
        if verbose:
            print(f"\n{'Alpha':<10} {'Weighted F-max':<18} {'S-min':<18} {'Avg Threshold':<18}")
            print("-" * 70)
            for res in results:
                print(f"{res['alpha']:<10.2f} {res['weighted_fmax']:<18.6f} "
                      f"{res['smin']:<18.6f} {res['avg_threshold']:<18.6f}")
            print("=" * 70)
        
        return results


# =========================================================================
# Utility Functions
# =========================================================================

def load_ia_weights(
    ia_path: str,
    go_to_idx: Dict[str, int],
    num_classes: int,
    default_weight: float = 0.0
) -> torch.Tensor:
    """
    IA.tsv 파일에서 IC 가중치 로드
    
    Args:
        ia_path: IA.tsv 파일 경로
        go_to_idx: GO term -> index 매핑
        num_classes: 전체 클래스 수
        default_weight: 기본 가중치 (매핑 없는 경우)
    
    Returns:
        ic_weights: (num_classes,) IC 가중치 텐서
    """
    ia_df = pd.read_csv(ia_path, sep='\t', header=None, names=['GO_Term', 'IC'])
    
    ic_weights = torch.full((num_classes,), default_weight, dtype=torch.float32)
    
    matched_count = 0
    for _, row in ia_df.iterrows():
        go_term = row['GO_Term']
        ic_value = row['IC']
        if go_term in go_to_idx:
            idx = go_to_idx[go_term]
            ic_weights[idx] = ic_value
            matched_count += 1
    
    print(f"✅ IC 가중치 로드 완료: {matched_count}/{len(ia_df)} GO terms 매칭됨")
    
    return ic_weights


# =========================================================================
# Azure ML Main Function
# =========================================================================

def setup_paths_for_azure(dataset_input_path: str = None, dataset_name: str = None, local_mode: bool = False):
    """
    Azure ML 또는 로컬 환경의 경로를 설정합니다.
    
    Args:
        dataset_input_path: Azure ML Job의 inputs.dataset 경로 (데이터셋 루트)
        dataset_name: 데이터셋 내부 폴더 이름 (예: "CAFA6_82404sq_Asymmetric Loss")
        local_mode: True면 로컬 경로 사용
    
    Returns:
        dict: 경로 정보 딕셔너리
    """
    if local_mode:
        # 로컬 모드
        base_dir = Path(__file__).parent.parent
        dataset_root = base_dir / "CAFA6_82404sq_Asymmetric Loss"
        cache_dir = base_dir / "validation_init"
        
        print("🏠 로컬 모드로 실행 중...")
        cafa6_root = dataset_root
    else:
        # Azure ML 모드 - train.py와 동일한 로직
        print("☁️ Azure ML 모드로 실행 중...")
        
        data_path = Path(dataset_input_path) if dataset_input_path else Path("./dataset")
        
        print("=" * 70)
        print("📂 데이터셋 경로 검증")
        print("=" * 70)
        print(f"🔍 data_path: {data_path}")
        print(f"🔍 dataset_name: {dataset_name}")
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
        # 1. data_path에 예상 폴더가 있는지 확인
        # 2. 있으면 data_path가 이미 데이터셋 루트
        # 3. 없으면 data_path/dataset_name이 데이터셋 루트
        
        expected_folders = ['embedded_fasta_for_1dcnn', 'go_basic', 'cafa-6-protein-function-prediction']
        has_expected_structure = any((data_path / folder).exists() for folder in expected_folders)
        
        if has_expected_structure:
            # data_path가 이미 데이터셋 루트
            cafa6_root = data_path
            print(f"\n✅ 감지: data_path가 이미 데이터셋 루트")
        else:
            # data_path/dataset_name이 데이터셋 루트일 가능성
            if dataset_name:
                potential_base = data_path / dataset_name
                if potential_base.exists() and any((potential_base / folder).exists() for folder in expected_folders):
                    cafa6_root = potential_base
                    print(f"\n✅ 감지: {dataset_name} 하위 폴더가 데이터셋 루트")
                else:
                    # 기본값: data_path 사용 (에러는 나중에 발생)
                    cafa6_root = data_path
                    print(f"\n⚠️ 예상 폴더 구조를 찾지 못함, data_path 사용")
            else:
                cafa6_root = data_path
                print(f"\n⚠️ dataset_name 미제공, data_path 사용")
        
        print(f"🔍 최종 cafa6_root: {cafa6_root}")
        
        # 캐시는 outputs에 저장 (Azure ML에서 자동으로 수집됨)
        cache_dir = Path("./validation_cache")
        cache_dir.mkdir(parents=True, exist_ok=True)
    
    # 경로 구성
    embed_dir = cafa6_root / "embedded_fasta_for_1dcnn"
    go_dir = cafa6_root / "go_basic"
    data_dir = cafa6_root / "cafa-6-protein-function-prediction"
    
    paths = {
        'dataset_root': cafa6_root,
        'embed_dir': embed_dir,
        'train_h5': embed_dir / "train" / "train_embeddings_82404sq.h5",
        'go_dir': go_dir,
        'go_mappings': go_dir / "go_term_mappings.pkl",
        'ancestor_matrix': go_dir / "go_ancestor_matrix.npz",
        'data_dir': data_dir,
        'cache_dir': cache_dir,
    }
    
    # 경로 검증 및 디버깅 정보
    print(f"\n📁 경로 검증 중...")
    for key, path in paths.items():
        if key in ['cache_dir']:  # 캐시는 자동 생성
            continue
        
        exists = path.exists()
        status = "✅" if exists else "❌"
        print(f"   {status} {key}: {path}")
        
        if not exists:
            # 디버깅: 상위 디렉토리 구조 출력
            parent = path.parent
            if parent.exists():
                print(f"      📂 상위 디렉토리 ({parent}) 내용:")
                try:
                    for item in list(parent.iterdir())[:10]:  # 최대 10개만
                        print(f"         - {item.name}")
                except Exception as e:
                    print(f"      ⚠️ 디렉토리 읽기 실패: {e}")
            
            raise FileNotFoundError(
                f"{key} 경로를 찾을 수 없습니다: {path}\n"
                f"데이터셋 마운트 경로를 확인하세요."
            )
    
    print(f"\n✅ 모든 경로 검증 완료")
    print(f"   데이터셋 루트: {paths['dataset_root']}")
    print(f"   HDF5 파일: {paths['train_h5']}")
    print(f"   GO 디렉토리: {paths['go_dir']}")
    print(f"   캐시 디렉토리: {paths['cache_dir']}")
    
    return paths


def main():
    """Azure ML Studio 또는 로컬에서 validation 실행"""
    
    # =========================================================================
    # 1. 커맨드라인 인자 파싱
    # =========================================================================
    parser = argparse.ArgumentParser(description='CAFA6 Model Validation')
    
    # 필수 인자
    parser.add_argument('--model_path', type=str, required=True,
                        help='모델 체크포인트 경로 (.pth 파일)')
    parser.add_argument('--output_dir', type=str, default='./outputs',
                        help='출력 디렉토리')
    
    # Azure ML 데이터셋 경로
    parser.add_argument('--dataset_path', type=str, default=None,
                        help='Azure ML 데이터셋 마운트 경로 (Job의 inputs.dataset 경로)')
    parser.add_argument('--dataset_name', type=str, default='CAFA6_82404sq_Asymmetric Loss',
                        help='데이터셋 내부 폴더 이름')
    
    # 선택 인자
    parser.add_argument('--local_mode', action='store_true',
                        help='로컬 모드 (Azure ML 외부 실행)')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='배치 크기')
    parser.add_argument('--val_split', type=float, default=0.1,
                        help='Validation 비율')
    parser.add_argument('--random_seed', type=int, default=42,
                        help='랜덤 시드')
    
    # Evaluation 옵션
    parser.add_argument('--skip_dynamic_threshold', action='store_true',
                        help='Dynamic Thresholding 평가 건너뛰기')
    parser.add_argument('--skip_ancestral_resonance', action='store_true',
                        help='Ancestral Resonance 평가 건너뛰기')
    parser.add_argument('--alpha_values', type=str, default='0.02,0.03,0.04,0.05',
                        help='Dynamic Thresholding alpha 값 (쉼표 구분)')
    parser.add_argument('--resonance_alphas', type=str, default='0.3,0.4,0.5,0.6,0.7',
                        help='Ancestral Resonance alpha 값 (쉼표 구분)')
    
    args = parser.parse_args()
    
    # Alpha 값 파싱
    alpha_values = [float(a) for a in args.alpha_values.split(',')]
    resonance_alphas = [float(a) for a in args.resonance_alphas.split(',')]
    
    # 출력 디렉토리 생성
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print("CAFA6 Validation Pipeline (Azure ML Version)")
    print("=" * 70)
    print(f"모델 경로: {args.model_path}")
    print(f"출력 디렉토리: {output_dir}")
    print(f"배치 크기: {args.batch_size}")
    print(f"Validation 비율: {args.val_split}")
    
    # =========================================================================
    # 2. MLflow 초기화
    # =========================================================================
    try:
        import mlflow
        mlflow.start_run()
        print("✅ MLflow 실행 시작")
        use_mlflow = True
    except Exception as e:
        print(f"⚠️ MLflow를 사용할 수 없습니다: {e}")
        use_mlflow = False
    
    # =========================================================================
    # 3. 경로 설정
    # =========================================================================
    paths = setup_paths_for_azure(
        dataset_input_path=args.dataset_path,
        dataset_name=args.dataset_name,
        local_mode=args.local_mode
    )
    
    # =========================================================================
    # 4. 디바이스 설정
    # =========================================================================
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n🖥️ 디바이스: {device}")
    if torch.cuda.is_available():
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    # =========================================================================
    # 5. 데이터 로드
    # =========================================================================
    from h5_data_loder import (
        H5MetadataAnalyzer,
        H5MultiDatasetAnalyzer,
        HDF5EmbeddingDataset,
        worker_init_fn,
        collate_fn_train,
    )
    from scipy.sparse import load_npz
    from sklearn.model_selection import train_test_split
    from torch.utils.data import DataLoader
    
    print("\n" + "=" * 70)
    print("📂 데이터 로드 중...")
    print("=" * 70)
    
    # HDF5 메타데이터
    MAX_SEQ_LENGTH = H5MetadataAnalyzer.DEFAULT_MAX_SEQ_LENGTH
    EMBEDDING_DIM = H5MetadataAnalyzer.DEFAULT_EMBEDDING_DIM
    
    multi_analyzer = H5MultiDatasetAnalyzer()
    train_analyzer = multi_analyzer.add_dataset('train', paths['train_h5'])
    
    train_metadata = train_analyzer.metadata
    train_ids = train_analyzer.ids
    train_offsets = train_analyzer.offsets
    train_lengths = train_analyzer.lengths
    
    print(f"✅ Train 데이터: {train_analyzer.num_sequences:,} 시퀀스")
    
    # GO Term 매핑
    with open(paths['go_mappings'], "rb") as f:
        go_mappings = pickle.load(f)
    
    term_to_idx = go_mappings['go_to_idx']
    idx_to_term = go_mappings['idx_to_go']
    num_go_terms = len(term_to_idx)
    
    print(f"✅ GO 매핑: {num_go_terms:,} terms")
    
    # Train terms
    train_terms_df = pd.read_csv(
        paths['data_dir'] / "Train" / "train_terms.tsv",
        sep="\t"
    )
    train_terms_df.columns = ["EntryID", "GO_Term", "Aspect"]
    train_terms_df = train_terms_df[train_terms_df['EntryID'].isin(set(train_ids))].copy()
    
    print(f"✅ Train terms: {len(train_terms_df):,} 레이블")
    
    # IC 가중치
    ic_weights = load_ia_weights(
        ia_path=str(paths['data_dir'] / 'IA.tsv'),
        go_to_idx=term_to_idx,
        num_classes=num_go_terms,
        default_weight=0.0
    )
    ic_weights_np = ic_weights.numpy()
    print(f"✅ IC 가중치: [{ic_weights_np.min():.4f}, {ic_weights_np.max():.4f}]")
    
    # Ancestor matrix
    ancestor_matrix = load_npz(paths['ancestor_matrix'])
    print(f"✅ Ancestor matrix: {ancestor_matrix.shape}")
    
    # =========================================================================
    # 6. ValidationEvaluator 초기화
    # =========================================================================
    print("\n" + "=" * 70)
    print("🔧 ValidationEvaluator 초기화")
    print("=" * 70)
    
    evaluator = ValidationEvaluator(
        train_ids=train_ids,
        term_to_idx=term_to_idx,
        ic_weights_np=ic_weights_np,
        cache_dir=str(paths['cache_dir']),
        device=device
    )
    
    # =========================================================================
    # 7. Label Matrix 준비
    # =========================================================================
    print("\n" + "=" * 70)
    print("🏷️ Label Matrix 준비")
    print("=" * 70)
    
    label_matrix_propagated = evaluator.prepare_label_matrix(
        train_terms_df=train_terms_df,
        ancestor_matrix=ancestor_matrix,
        force_rebuild=False
    )
    
    print(f"✅ Label Matrix: {label_matrix_propagated.shape}")
    print(f"   Positive labels: {label_matrix_propagated.sum():.0f}")
    
    # =========================================================================
    # 8. Train/Validation Split
    # =========================================================================
    print("\n" + "=" * 70)
    print("🔄 Train/Validation Split")
    print("=" * 70)
    
    num_train_samples = train_metadata['num_sequences']
    train_idx, val_idx = train_test_split(
        np.arange(num_train_samples),
        test_size=args.val_split,
        random_state=args.random_seed
    )
    
    y_val = label_matrix_propagated[val_idx]
    
    print(f"Train: {len(train_idx):,}, Validation: {len(val_idx):,}")
    
    # Validation DataLoader
    val_dataset = HDF5EmbeddingDataset(
        h5_path=paths['train_h5'],
        indices=val_idx,
        labels=y_val,
        offsets=train_offsets,
        lengths=train_lengths,
        max_length=MAX_SEQ_LENGTH,
        pad=False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_fn_train,
        persistent_workers=False
    )
    
    print(f"✅ Validation DataLoader 생성 완료")
    
    # =========================================================================
    # 9. 모델 로드
    # =========================================================================
    from model import Hierarchical1DCNNClassifier, ModelConfig, predict
    
    print("\n" + "=" * 70)
    print("📥 모델 로드")
    print("=" * 70)
    print(f"모델 경로: {args.model_path}")
    
    checkpoint = torch.load(args.model_path, map_location=device)
    
    if 'model_config' in checkpoint:
        model_config = ModelConfig.from_dict(checkpoint['model_config'])
        print(f"✅ Model Config 로드")
    else:
        print("⚠️ Checkpoint에 config 없음, 기본값 사용")
        model_config = ModelConfig(
            embedding_dim=EMBEDDING_DIM,
            num_classes=num_go_terms,
            conv_channels=[512, 1024, 2048],
            kernel_sizes=[7, 5, 3],
            fc_dims=[2048],
            dropout=0.3,
            use_residual=True,
        )
    
    model = Hierarchical1DCNNClassifier.from_config(model_config).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"✅ 모델 로드 완료")
    if 'epoch' in checkpoint:
        print(f"   Epoch: {checkpoint['epoch']}")
    if 'train_loss' in checkpoint:
        print(f"   Train Loss: {checkpoint['train_loss']:.4f}")
    if 'val_loss' in checkpoint:
        print(f"   Val Loss: {checkpoint['val_loss']:.4f}")
    
    # =========================================================================
    # 10. Validation 예측
    # =========================================================================
    print("\n" + "=" * 70)
    print("🔮 Validation 예측")
    print("=" * 70)
    
    val_predictions = predict(
        model=model,
        dataloader=val_loader,
        device=device,
        use_amp=torch.cuda.is_available(),
        non_blocking=True,
        return_tensor=False
    )
    
    print(f"✅ 예측 완료: {val_predictions.shape}")
    
    # =========================================================================
    # 11. CAFA6 평가
    # =========================================================================
    print("\n" + "=" * 70)
    print("📊 CAFA6 평가 (Weighted F-max & S-min)")
    print("=" * 70)
    
    cafa6_results = evaluator.evaluate_cafa6(
        predictions=val_predictions,
        labels=y_val,
        thresholds=None,
        verbose=True
    )
    
    print(f"\n✅ CAFA6 결과:")
    print(f"   Weighted F-max: {cafa6_results['fmax']:.6f} @ {cafa6_results['fmax_threshold']:.4f}")
    print(f"   S-min: {cafa6_results['smin']:.6f} @ {cafa6_results['smin_threshold']:.4f}")
    
    # MLflow 로깅
    if use_mlflow:
        try:
            mlflow.log_metrics({
                'val_fmax': cafa6_results['fmax'],
                'val_smin': cafa6_results['smin'],
                'val_fmax_threshold': cafa6_results['fmax_threshold'],
                'val_smin_threshold': cafa6_results['smin_threshold'],
            })
            print("✅ MLflow에 CAFA6 메트릭 로깅 완료")
        except Exception as e:
            print(f"⚠️ MLflow 로깅 실패: {e}")
    
    # 결과 저장
    cafa6_df = pd.DataFrame([{
        'fmax': cafa6_results['fmax'],
        'smin': cafa6_results['smin'],
        'fmax_threshold': cafa6_results['fmax_threshold'],
        'smin_threshold': cafa6_results['smin_threshold'],
    }])
    cafa6_df.to_csv(output_dir / 'cafa6_results.csv', index=False)
    print(f"💾 CAFA6 결과 저장: {output_dir / 'cafa6_results.csv'}")
    
    # =========================================================================
    # 12. Dynamic Thresholding (선택)
    # =========================================================================
    if not args.skip_dynamic_threshold:
        print("\n" + "=" * 70)
        print("🎯 IC-Adaptive Dynamic Thresholding")
        print("=" * 70)
        
        dynamic_results = evaluator.dynamic_thresholding(
            predictions=val_predictions,
            labels=y_val,
            base_threshold=cafa6_results['fmax_threshold'],
            alpha_values=alpha_values,
            use_gpu=True,
            show_progress=True,
            verbose=True
        )
        
        best_dynamic = max(dynamic_results, key=lambda x: x['weighted_fmax'])
        
        print(f"\n✅ 최적 Dynamic Threshold:")
        print(f"   Alpha: {best_dynamic['alpha']:.2f}")
        print(f"   F-max: {best_dynamic['weighted_fmax']:.6f}")
        print(f"   개선: {best_dynamic['weighted_fmax'] - cafa6_results['fmax']:+.6f}")
        
        # MLflow 로깅
        if use_mlflow:
            try:
                mlflow.log_metrics({
                    'val_dynamic_fmax': best_dynamic['weighted_fmax'],
                    'val_dynamic_alpha': best_dynamic['alpha'],
                    'val_dynamic_improvement': best_dynamic['weighted_fmax'] - cafa6_results['fmax'],
                })
            except Exception as e:
                print(f"⚠️ MLflow 로깅 실패: {e}")
        
        # 결과 저장
        dynamic_df = pd.DataFrame(dynamic_results)
        dynamic_df.to_csv(output_dir / 'dynamic_threshold_results.csv', index=False)
        print(f"💾 Dynamic Threshold 결과 저장")
    
    # =========================================================================
    # 13. Ancestral Resonance (선택)
    # =========================================================================
    if not args.skip_ancestral_resonance:
        print("\n" + "=" * 70)
        print("🔬 Ancestral Resonance Alpha 최적화")
        print("=" * 70)
        
        resonance_results = []
        
        for alpha_test in tqdm(resonance_alphas, desc="Alpha 테스트"):
            # Resonance 적용
            resonated_preds = evaluator.apply_ancestral_resonance(
                predictions=val_predictions,
                ancestor_matrix_sparse=ancestor_matrix,
                alpha=alpha_test,
                use_gpu=True,
                show_progress=False
            )
            
            # CAFA6 평가
            result = evaluator.evaluate_cafa6(
                predictions=resonated_preds,
                labels=y_val,
                use_gpu=True,
                show_progress=False,
                verbose=False
            )
            
            resonance_results.append({
                'alpha': alpha_test,
                'fmax': result['fmax'],
                'smin': result['smin'],
                'fmax_threshold': result['fmax_threshold'],
            })
        
        # 최적 alpha
        best_resonance = max(resonance_results, key=lambda x: x['fmax'])
        
        print(f"\n✅ 최적 Ancestral Resonance:")
        print(f"   Alpha: {best_resonance['alpha']:.1f}")
        print(f"   F-max: {best_resonance['fmax']:.6f}")
        print(f"   개선: {best_resonance['fmax'] - cafa6_results['fmax']:+.6f}")
        
        # MLflow 로깅
        if use_mlflow:
            try:
                mlflow.log_metrics({
                    'val_resonance_fmax': best_resonance['fmax'],
                    'val_resonance_alpha': best_resonance['alpha'],
                    'val_resonance_improvement': best_resonance['fmax'] - cafa6_results['fmax'],
                })
            except Exception as e:
                print(f"⚠️ MLflow 로깅 실패: {e}")
        
        # 결과 저장
        resonance_df = pd.DataFrame(resonance_results)
        resonance_df.to_csv(output_dir / 'ancestral_resonance_results.csv', index=False)
        print(f"💾 Ancestral Resonance 결과 저장")
    
    # =========================================================================
    # 14. 최종 요약
    # =========================================================================
    print("\n" + "=" * 70)
    print("✅ Validation 완료!")
    print("=" * 70)
    print(f"📊 CAFA6 Weighted F-max: {cafa6_results['fmax']:.6f}")
    print(f"📊 CAFA6 S-min: {cafa6_results['smin']:.6f}")
    
    if not args.skip_dynamic_threshold:
        print(f"🎯 Dynamic F-max: {best_dynamic['weighted_fmax']:.6f} (α={best_dynamic['alpha']:.2f})")
    
    if not args.skip_ancestral_resonance:
        print(f"🔬 Resonance F-max: {best_resonance['fmax']:.6f} (α={best_resonance['alpha']:.1f})")
    
    print(f"\n💾 결과 저장 위치: {output_dir}")
    
    # MLflow 종료
    if use_mlflow:
        try:
            mlflow.end_run()
            print("✅ MLflow 실행 종료")
        except Exception as e:
            print(f"⚠️ MLflow 종료 실패: {e}")
    
    print("\n🎉 모든 작업이 완료되었습니다!")


if __name__ == "__main__":
    main()
