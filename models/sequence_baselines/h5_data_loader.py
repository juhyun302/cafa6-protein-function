"""
HDF5 임베딩 데이터 로더 모듈 (1D CNN용)

HDF5 파일 구조 (ESM2_tokenizer.ipynb에서 생성):
├── embeddings (N_total_tokens, 1280) float16  # 모든 토큰 연속 저장
├── offsets (num_sequences,) int64             # 각 시퀀스 시작 위치
├── lengths (num_sequences,) int32             # 각 시퀀스 실제 길이
└── ids (num_sequences,) string                # 단백질 ID

임베딩 접근 방법:
    # i번째 시퀀스 임베딩 가져오기
    start = offsets[i]
    length = lengths[i]
    embedding = embeddings[start:start+length]  # (length, 1280)

1D CNN은 토큰별 임베딩을 사용하므로 평균 풀링을 적용하지 않음
실제 임베딩은 Dataset에서 on-demand로 로드
"""

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import Dict, Optional, Union, List, Tuple

class H5MetadataAnalyzer:
    """
    HDF5 임베딩 파일의 메타데이터를 분석하는 클래스

    Attributes:
        h5_path: HDF5 파일 경로
        metadata: 로드된 메타데이터 딕셔너리
        ids_parsed: 파싱된 UniProt Accession ID 배열
    """

    # 1D CNN 기본 설정
    DEFAULT_MAX_SEQ_LENGTH = 1024  # 최대 시퀀스 길이 (패딩/자르기용)
    DEFAULT_EMBEDDING_DIM = 1280   # ESM2-650M 임베딩 차원

    def __init__(self, h5_path: Optional[Union[str, Path]] = None):
        """
        H5MetadataAnalyzer 초기화

        Args:
            h5_path: HDF5 파일 경로 (선택적, 나중에 load()로 로드 가능)
        """
        self.h5_path: Optional[Path] = None
        self.metadata: Optional[Dict] = None
        self.ids_parsed: Optional[np.ndarray] = None

        if h5_path is not None:
            self.load(h5_path)

    def load(self, h5_path: Union[str, Path]) -> Dict:
        """
        HDF5 파일에서 메타데이터만 로드 (1D CNN용)
        실제 임베딩은 Dataset에서 on-demand로 로드

        Args:
            h5_path: HDF5 파일 경로

        Returns:
            metadata: dict containing offsets, lengths, ids, num_sequences, embedding_dim
        """
        self.h5_path = Path(h5_path)

        print(f"HDF5 메타데이터 로드 중: {self.h5_path}")

        with h5py.File(self.h5_path, 'r') as f:
            # 메타데이터 확인
            num_sequences = f.attrs['num_sequences']
            total_tokens = f.attrs['total_tokens']
            embedding_dim = f.attrs['embedding_dim']

            print(f"   시퀀스 수: {num_sequences:,}")
            print(f"   총 토큰 수: {total_tokens:,}")
            print(f"   임베딩 차원: {embedding_dim}")
            print(f"   평균 시퀀스 길이: {total_tokens / num_sequences:.1f}")

            # 메타데이터 로드
            offsets = f['offsets'][:]
            lengths = f['lengths'][:]
            ids = f['ids'][:]

        self.metadata = {
            'offsets': offsets,
            'lengths': lengths,
            'ids': ids,
            'num_sequences': num_sequences,
            'total_tokens': total_tokens,
            'embedding_dim': embedding_dim
        }

        # ID 파싱
        self.ids_parsed = np.array([self.extract_uniprot_accession(x) for x in ids])

        print(f"   메타데이터 로드 완료")
        return self.metadata

    @staticmethod
    def extract_uniprot_accession(full_id: Union[str, bytes]) -> str:
        """
        UniProt 전체 ID에서 Accession만 추출

        Args:
            full_id: 전체 UniProt ID (예: 'sp|A0A0C5B5G6|MOTSC_HUMAN')

        Returns:
            Accession만 추출된 ID (예: 'A0A0C5B5G6')
        """
        if isinstance(full_id, bytes):
            full_id = full_id.decode('utf-8')
        if '|' in full_id:
            parts = full_id.split('|')
            return parts[1]  # sp|ACCESSION|NAME 형식에서 ACCESSION 추출
        return full_id  # 이미 Accession만 있는 경우

    @property
    def offsets(self) -> Optional[np.ndarray]:
        """각 시퀀스 시작 위치 배열"""
        return self.metadata['offsets'] if self.metadata else None

    @property
    def lengths(self) -> Optional[np.ndarray]:
        """각 시퀀스 실제 길이 배열"""
        return self.metadata['lengths'] if self.metadata else None

    @property
    def ids_raw(self) -> Optional[np.ndarray]:
        """원본 ID 배열"""
        return self.metadata['ids'] if self.metadata else None

    @property
    def ids(self) -> Optional[np.ndarray]:
        """파싱된 UniProt Accession ID 배열"""
        return self.ids_parsed

    @property
    def num_sequences(self) -> Optional[int]:
        """시퀀스 총 개수"""
        return self.metadata['num_sequences'] if self.metadata else None

    @property
    def total_tokens(self) -> Optional[int]:
        """총 토큰 수"""
        return self.metadata['total_tokens'] if self.metadata else None

    @property
    def embedding_dim(self) -> Optional[int]:
        """임베딩 차원"""
        return self.metadata['embedding_dim'] if self.metadata else None

    def get_length_statistics(self) -> Dict:
        """
        시퀀스 길이 분포 통계 반환

        Returns:
            dict: min, max, mean, std, median 통계
        """
        if self.metadata is None:
            raise ValueError("메타데이터가 로드되지 않았습니다. 먼저 load()를 호출하세요.")

        lengths = self.metadata['lengths']
        return {
            'min': int(lengths.min()),
            'max': int(lengths.max()),
            'mean': float(lengths.mean()),
            'std': float(lengths.std()),
            'median': float(np.median(lengths))
        }

    def print_summary(self) -> None:
        """메타데이터 요약 정보 출력"""
        if self.metadata is None:
            print("메타데이터가 로드되지 않았습니다.")
            return

        print("\n" + "=" * 70)
        print("HDF5 메타데이터 요약")
        print("=" * 70)
        print(f"파일 경로: {self.h5_path}")
        print(f"시퀀스 수: {self.num_sequences:,}")
        print(f"총 토큰 수: {self.total_tokens:,}")
        print(f"임베딩 차원: {self.embedding_dim}")

        stats = self.get_length_statistics()
        print(f"\n시퀀스 길이 분포:")
        print(f"  - min: {stats['min']}")
        print(f"  - max: {stats['max']}")
        print(f"  - mean: {stats['mean']:.1f}")
        print(f"  - std: {stats['std']:.1f}")
        print(f"  - median: {stats['median']:.1f}")

        print(f"\n🔍 ID 샘플:")
        print(f"  원본 ID: {self.ids_raw[0]}")
        print(f"  파싱된 ID: {self.ids_parsed[0]}")
        print("=" * 70)

    def load_embedding_by_index(self, idx: int) -> Tuple[np.ndarray, str]:
        """
        HDF5에서 특정 인덱스의 임베딩을 로드

        Args:
            idx: 시퀀스 인덱스

        Returns:
            embedding: (length, 1280) float16 배열
            protein_id: 단백질 ID (파싱된 Accession)
        """
        if self.metadata is None:
            raise ValueError("메타데이터가 로드되지 않았습니다. 먼저 load()를 호출하세요.")

        with h5py.File(self.h5_path, 'r') as f:
            offset = self.offsets[idx]
            length = self.lengths[idx]
            # 연속 저장된 embeddings에서 offset부터 length만큼 슬라이싱
            embedding = f['embeddings'][offset:offset+length]

        protein_id = self.ids_parsed[idx]
        return embedding, protein_id

    def load_embedding_by_id(self, protein_id: str) -> np.ndarray:
        """
        HDF5에서 단백질 ID로 임베딩을 로드

        Args:
            protein_id: 단백질 ID (Accession, 예: 'A0A0C5B5G6')

        Returns:
            embedding: (length, 1280) float16 배열
        """
        if self.metadata is None:
            raise ValueError("메타데이터가 로드되지 않았습니다. 먼저 load()를 호출하세요.")

        # 파싱된 ID에서 검색
        idx_matches = np.where(self.ids_parsed == protein_id)[0]
        if len(idx_matches) == 0:
            raise ValueError(f"Protein ID '{protein_id}' not found")

        idx = idx_matches[0]
        embedding, _ = self.load_embedding_by_index(idx)
        return embedding

    def get_id_to_index_map(self) -> Dict[str, int]:
        """
        단백질 ID -> 인덱스 매핑 딕셔너리 반환

        Returns:
            dict: {protein_id: index}
        """
        if self.ids_parsed is None:
            raise ValueError("메타데이터가 로드되지 않았습니다.")
        return {pid: idx for idx, pid in enumerate(self.ids_parsed)}


class H5MultiDatasetAnalyzer:
    """
    여러 HDF5 파일(Train/Test)을 함께 분석하는 클래스
    """

    def __init__(self):
        """H5MultiDatasetAnalyzer 초기화"""
        self.datasets: Dict[str, H5MetadataAnalyzer] = {}

    def add_dataset(self, name: str, h5_path: Union[str, Path]) -> H5MetadataAnalyzer:
        """
        데이터셋 추가

        Args:
            name: 데이터셋 이름 (예: 'train', 'test')
            h5_path: HDF5 파일 경로

        Returns:
            H5MetadataAnalyzer: 로드된 분석기 인스턴스
        """
        analyzer = H5MetadataAnalyzer(h5_path)
        self.datasets[name] = analyzer
        return analyzer

    def get_dataset(self, name: str) -> Optional[H5MetadataAnalyzer]:
        """데이터셋 분석기 반환"""
        return self.datasets.get(name)

    def print_all_summaries(self) -> None:
        """모든 데이터셋 요약 정보 출력"""
        print("\n" + "=" * 70)
        print(" HDF5 메타데이터 로드 완료")
        print("=" * 70)

        for name, analyzer in self.datasets.items():
            print(f"\n[{name.upper()}]")
            print(f"  시퀀스 수: {analyzer.num_sequences:,}")
            print(f"  총 토큰 수: {analyzer.total_tokens:,}")
            stats = analyzer.get_length_statistics()
            print(f"  길이 분포 - min: {stats['min']}, max: {stats['max']}, mean: {stats['mean']:.1f}")

        print("\n" + "=" * 70)

    def compare_datasets(self) -> Dict:
        """
        데이터셋 비교 통계 반환

        Returns:
            dict: 각 데이터셋별 주요 통계
        """
        comparison = {}
        for name, analyzer in self.datasets.items():
            stats = analyzer.get_length_statistics()
            comparison[name] = {
                'num_sequences': analyzer.num_sequences,
                'total_tokens': analyzer.total_tokens,
                'embedding_dim': analyzer.embedding_dim,
                'length_stats': stats
            }
        return comparison


# =========================================================================
# HDF5EmbeddingDataset - PyTorch Dataset
# =========================================================================

class HDF5EmbeddingDataset(Dataset):
    """
    HDF5 임베딩 파일을 PyTorch Dataset으로 래핑 (1D-CNN 학습용 최적화)

    특징:
    - Lazy loading: 필요할 때만 HDF5에서 임베딩 로드
    - Multiprocessing 지원: worker_init_fn과 함께 사용 시 각 워커별 파일 핸들 관리
    - Dynamic Padding 지원: pad=False로 설정하고 collate_fn에서 패딩 처리
    - Subset 지원: indices로 전체 데이터 중 일부만 사용 가능 (Train/Val split)

    Args:
        h5_path: HDF5 파일 경로
        indices: 사용할 시퀀스 인덱스 배열 (None이면 전체 사용)
        labels: 라벨 매트릭스 (None이면 라벨 없음)
        offsets: 각 시퀀스 시작 오프셋 (None이면 파일에서 로드)
        lengths: 각 시퀀스 실제 길이 (None이면 파일에서 로드)
        max_length: 최대 시퀀스 길이 (자르기용)
        padding_value: 패딩 값
        pad: __getitem__에서 패딩 수행 여부 (False면 dynamic padding용 원본 반환)
    """

    def __init__(self, h5_path: Union[str, Path],
                 indices: Optional[np.ndarray] = None,
                 labels: Optional[np.ndarray] = None,
                 offsets: Optional[np.ndarray] = None,
                 lengths: Optional[np.ndarray] = None,
                 max_length: int = 1024,
                 padding_value: float = 0.0,
                 pad: bool = True):

        self.h5_path = Path(h5_path)
        self.max_length = max_length
        self.padding_value = padding_value
        self.pad = pad

        self.indices = indices
        self.labels = labels

        # HDF5 파일 핸들 (lazy loading)
        self._h5_file = None

        # 메타데이터 설정 (외부 주입 또는 파일 로드)
        if offsets is not None and lengths is not None:
            self.offsets = offsets
            self.lengths = lengths
            self.num_sequences = len(lengths)
            # embedding_dim은 나중에 로드하거나 기본값 사용
            self.embedding_dim = 1280
            self.ids = None # 외부 주입시 보통 ID 필요 없음
        else:
            # 메타데이터 로드 (파일 열지 않고 메모리에 캐시)
            with h5py.File(self.h5_path, 'r') as f:
                self.num_sequences = f.attrs['num_sequences']
                self.embedding_dim = f.attrs['embedding_dim']
                self.offsets = f['offsets'][:]
                self.lengths = f['lengths'][:]
                self.ids = f['ids'][:]

            # ID 파싱 (UniProt Accession 추출)
            self.ids_parsed = np.array([
                H5MetadataAnalyzer.extract_uniprot_accession(x) for x in self.ids
            ])

        print(f"✅ HDF5 데이터셋 로드: {len(self)}개 시퀀스 (Pad={pad})")

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

    def __len__(self) -> int:
        return len(self.indices) if self.indices is not None else self.num_sequences

    def __getitem__(self, idx: int) -> Dict:
        # 실제 인덱스 (indices가 있으면 매핑, 없으면 그대로)
        real_idx = self.indices[idx] if self.indices is not None else idx

        # HDF5에서 임베딩 로드
        f = self._get_h5_file()
        start = self.offsets[real_idx]
        length = self.lengths[real_idx]

        # 실제 사용할 길이 (max_length 이하로 제한)
        actual_length = min(length, self.max_length)

        # 패딩 없이 실제 길이만큼만 로드 (메모리 절약)
        embedding = f['embeddings'][start:start+actual_length].astype(np.float32)

        if self.pad:
            # 1. Static Padding Mode (기본)
            padded = np.full((self.max_length, self.embedding_dim),
                             self.padding_value, dtype=np.float32)
            padded[:actual_length] = embedding

            attention_mask = np.zeros(self.max_length, dtype=np.float32)
            attention_mask[:actual_length] = 1.0

            result = {
                'embedding': torch.from_numpy(padded).float(),
                'attention_mask': torch.from_numpy(attention_mask),
                'length': actual_length,
            }
            # ID가 로드되어 있다면 포함
            if hasattr(self, 'ids_parsed'):
                 result['id'] = self.ids_parsed[real_idx]

        else:
            # 2. Dynamic Padding Mode (collate_fn용 원본 반환)
            result = {
                'embedding': torch.from_numpy(embedding).float(),
                'length': actual_length
            }

        # 라벨이 있으면 추가
        if self.labels is not None:
            result['label'] = torch.from_numpy(self.labels[idx]).float()

        return result

    def __del__(self):
        if self._h5_file is not None:
            self._h5_file.close()

    # =========================================================================
    # 멀티프로세싱 지원 (pickle 직렬화)
    # =========================================================================
    # DataLoader의 num_workers > 0일 때 각 워커 프로세스에서
    # HDF5 파일 핸들을 별도로 열기 위해 필요합니다.
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


# =========================================================================
# 유틸리티 함수 (DataLoader용)
# =========================================================================

def worker_init_fn(worker_id):
    """
    DataLoader 워커 초기화 함수
    각 워커에서 HDF5 파일 핸들을 별도로 열기 위해 사용
    """
    worker_info = torch.utils.data.get_worker_info()
    if worker_info is not None:
        dataset = worker_info.dataset
        # HDF5 파일 핸들 리셋 (각 워커에서 새로 열기)
        if hasattr(dataset, 'reset_file_handle'):
            dataset.reset_file_handle()

def collate_fn_train(batch):
    """
    학습용 collate_fn: 배치 내 동적 패딩

    Args:
        batch: list of {'embedding': tensor, 'length': int, 'label': tensor}

    Returns:
        dict with padded 'embedding', 'attention_mask', 'label'
    """
    # 배치 내 최대 길이 계산
    lengths = [item['length'] for item in batch]
    batch_max_len = max(lengths)

    batch_size = len(batch)
    embedding_dim = batch[0]['embedding'].shape[1]

    # 배치 내 최대 길이로 패딩 (메모리 최적화)
    padded_embeddings = torch.zeros(batch_size, batch_max_len, embedding_dim)
    attention_masks = torch.zeros(batch_size, batch_max_len)

    has_label = 'label' in batch[0]
    if has_label:
        labels = torch.stack([item['label'] for item in batch])

    for i, item in enumerate(batch):
        length = item['length']
        padded_embeddings[i, :length] = item['embedding']
        attention_masks[i, :length] = 1.0

    result = {
        'embedding': padded_embeddings,
        'attention_mask': attention_masks,
        'length': torch.tensor(lengths),
    }

    if has_label:
        result['label'] = labels

    return result

def collate_fn_test(batch):
    """
    테스트용 collate_fn: 배치 내 동적 패딩 (라벨 없음)
    """
    return collate_fn_train(batch)


# =========================================================================
# 메인 실행 예제
# =========================================================================

if __name__ == "__main__":
    # 예제 사용법
    print("=" * 70)
    print("H5 데이터 로더 모듈 사용 예제")
    print("=" * 70)

    # 기본 경로 설정 (필요시 수정)
    BASE_DIR = Path(__file__).parent.parent
    EMBED_DIR = BASE_DIR / "embedded_fasta_for_1dcnn"
    TRAIN_H5_PATH = EMBED_DIR / "train" / "train_embeddings.h5"

    # 방법 3: PyTorch Dataset 사용
    print("\n[방법 3] HDF5EmbeddingDataset (1D-CNN 학습용):")
    if TRAIN_H5_PATH.exists():
        # 기본 모드 (Static Padding)
        dataset = HDF5EmbeddingDataset(TRAIN_H5_PATH, max_length=1024, pad=True)
        sample = dataset[0]

        print(f"  Dataset 크기: {len(dataset)}")
        print(f"  샘플 데이터 (Static Padding):")
        print(f"    embedding shape: {sample['embedding'].shape} (max_length, embedding_dim)")
        print(f"    actual length: {sample['length']}")

        # Dynamic Padding 모드
        dataset_dyn = HDF5EmbeddingDataset(TRAIN_H5_PATH, max_length=1024, pad=False)
        sample_dyn = dataset_dyn[0]
        print(f"  샘플 데이터 (Dynamic Padding - Raw):")
        print(f"    embedding shape: {sample_dyn['embedding'].shape} (actual_length, embedding_dim)")
