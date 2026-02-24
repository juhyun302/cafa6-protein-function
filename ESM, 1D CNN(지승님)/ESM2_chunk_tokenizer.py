"""
ESM2-3B 청크 기반 토큰화 및 임베딩 추출

- 길이 1022 초과 시 슬라이딩 윈도우로 분할
- 각 청크를 mean pooling하여 (청크 개수, 2560) 형태로 저장
- 겹치는 구간 평균화 없이 청크 그대로 보존
"""

from __future__ import annotations

import os
from typing import List, Tuple
from multiprocessing import Pool, cpu_count
from functools import partial

import h5py
import numpy as np
import torch
import torch.nn as nn
from Bio import SeqIO
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

# ESM2-3B 모델 설정
MODEL_NAME = "facebook/esm2_t36_3B_UR50D"
EMBEDDING_DIM = 2560

# 토큰화/슬라이딩 윈도우 설정
MAX_LEN = 1022  # (1024 - CLS/EOS)
STRIDE = 512    # 50% overlap
WINDOW_BATCH_SIZE = 4  # 3B 모델은 메모리가 더 필요하므로 배치 크기 감소
NUM_WORKERS = min(8, cpu_count())  # 멀티프로세싱 워커 수 (기본값: 8 또는 CPU 코어 수 중 작은 값)


def load_model_and_tokenizer(model_name: str = MODEL_NAME, use_fp16: bool = True):
    """
    모델과 토크나이저를 로드.
    
    Args:
        model_name: 모델 이름
        use_fp16: True면 FP16 정밀도로 로드 (메모리 최적화, GPU만 지원)
    
    Returns:
        tokenizer, model, device
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # FP16은 GPU에서만 지원
    if use_fp16 and device.type == "cuda":
        print("💾 FP16 정밀도로 모델 로드 중... (메모리 최적화)")
        model = AutoModel.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",  # 자동으로 GPU에 배치
        )
        print("✅ FP16 모델 로드 완료")
    else:
        if use_fp16 and device.type == "cpu":
            print("⚠️ CPU 환경에서는 FP16을 사용할 수 없습니다. FP32로 로드합니다.")
        model = AutoModel.from_pretrained(model_name)
        model = model.to(device)
    
    # DataParallel은 device_map="auto"와 함께 사용할 수 없으므로 조건부 적용
    if device.type == "cuda" and torch.cuda.device_count() > 1 and not use_fp16:
        model = nn.DataParallel(model)
    
    model.eval()
    return tokenizer, model, device


def chunk_sequence(seq: str, max_len: int = MAX_LEN, stride: int = STRIDE) -> List[Tuple[int, int, str]]:
    """
    슬라이딩 윈도우로 시퀀스를 분할.
    
    Returns:
        List of (start_pos, end_pos, chunk_seq)
    """
    seq_len = len(seq)
    
    if seq_len <= max_len:
        return [(0, seq_len, seq)]

    windows: List[Tuple[int, int, str]] = []
    start = 0
    
    while True:
        end = min(start + max_len, seq_len)
        windows.append((start, end, seq[start:end]))
        
        if end == seq_len:
            break
            
        start += stride
        
        # 마지막 청크가 서열 끝을 포함하도록 조정
        if start + max_len >= seq_len:
            start = seq_len - max_len
    
    return windows


def embed_chunks(
    sequence: str,
    tokenizer,
    model,
    device: torch.device,
    max_len: int = MAX_LEN,
    stride: int = STRIDE,
    window_batch_size: int = WINDOW_BATCH_SIZE,
    use_amp: bool | None = None,
) -> torch.Tensor:
    """
    슬라이딩 윈도우로 청크를 생성하고 각 청크를 mean pooling하여 반환.

    Returns:
        torch.Tensor: (num_chunks, EMBEDDING_DIM) on CPU (FP32)
    """
    windows = chunk_sequence(sequence, max_len=max_len, stride=stride)
    num_chunks = len(windows)

    chunk_embeddings = torch.zeros((num_chunks, EMBEDDING_DIM), dtype=torch.float32)

    if use_amp is None:
        use_amp = device.type == "cuda"

    for batch_start in range(0, num_chunks, window_batch_size):
        batch_end = min(batch_start + window_batch_size, num_chunks)
        batch = windows[batch_start:batch_end]
        batch_seqs = [w[2] for w in batch]  # chunk sequences
        chunk_lens = [len(s) for s in batch_seqs]

        tokenized = tokenizer(
            batch_seqs,
            return_tensors="pt",
            padding=True,
            truncation=False,
            add_special_tokens=True,
        )
        tokenized = {k: v.to(device) for k, v in tokenized.items()}

        with torch.no_grad():
            # FP16 모델의 경우 autocast는 불필요하지만 안전을 위해 유지
            with torch.autocast(device_type=device.type, enabled=use_amp, dtype=torch.float16 if use_amp else None):
                outputs = model(**tokenized)
                hidden = outputs.last_hidden_state  # (batch, L+2, dim)

        for i, chunk_len in enumerate(chunk_lens):
            # CLS/EOS 제외하고 AA 토큰만 추출
            chunk_emb = hidden[i, 1:chunk_len + 1, :]  # (chunk_len, dim)
            # Mean pooling (FP16이면 자동으로 FP32로 변환)
            mean_emb = chunk_emb.mean(dim=0)  # (dim,)
            # FP32로 변환하여 저장 (정확도 유지)
            chunk_embeddings[batch_start + i] = mean_emb.cpu().float()

    return chunk_embeddings


def _calculate_chunk_counts(args):
    """시퀀스의 청크 개수 계산 (멀티프로세싱용)"""
    seq, max_len, stride = args
    windows = chunk_sequence(seq, max_len=max_len, stride=stride)
    return len(windows)
    """시퀀스의 청크 개수 계산 (멀티프로세싱용)"""
    windows = chunk_sequence(seq, max_len=max_len, stride=stride)
    return len(windows)


def extract_chunk_embeddings(
    fasta_path: str,
    save_path: str,
    max_len: int = MAX_LEN,
    stride: int = STRIDE,
    window_batch_size: int = WINDOW_BATCH_SIZE,
    model_name: str = MODEL_NAME,
    limit: int | None = None,
    use_fp16: bool = True,
    num_workers: int = NUM_WORKERS,
):
    """
    FASTA → 청크 기반 임베딩 추출 → HDF5 저장.
    
    Args:
        fasta_path: 입력 FASTA 파일 경로
        save_path: 출력 HDF5 파일 경로
        max_len: 최대 시퀀스 길이
        stride: 슬라이딩 윈도우 stride
        window_batch_size: 청크 배치 크기
        model_name: 모델 이름
        limit: 처리할 시퀀스 개수 제한 (None이면 전체)
        use_fp16: FP16 정밀도 사용 여부 (메모리 최적화)
        num_workers: 멀티프로세싱 워커 수 (기본값: 8)
    
    저장 형식:
        - embeddings: (total_chunks, EMBEDDING_DIM) - 모든 청크 임베딩
        - chunk_offsets: 각 시퀀스의 청크 시작 인덱스
        - chunk_counts: 각 시퀀스의 청크 개수
        - ids: 단백질 ID 목록
    """
    tokenizer, model, device = load_model_and_tokenizer(model_name, use_fp16=use_fp16)

    # FASTA 파일 파싱 (순차 처리 - 파일 I/O는 병렬화 불가)
    print(f"📖 FASTA 파일 파싱 중...")
    ids: List[str] = []
    sequences: List[str] = []
    for i, record in enumerate(SeqIO.parse(fasta_path, "fasta")):
        if limit is not None and i >= limit:
            break
        ids.append(record.id)
        sequences.append(str(record.seq))

    num_sequences = len(sequences)
    print(f"✅ 파싱 완료: {num_sequences}개 시퀀스")
    
    # 각 시퀀스의 청크 개수 미리 계산 (멀티프로세싱으로 병렬화)
    print(f"🔢 청크 개수 계산 중... (워커 수: {num_workers})")
    if num_workers > 1 and num_sequences > 50:
        # 멀티프로세싱으로 병렬 계산
        chunk_args = [(seq, max_len, stride) for seq in sequences]
        with Pool(processes=num_workers) as pool:
            chunk_counts = pool.map(_calculate_chunk_counts, chunk_args)
        print(f"✅ 병렬 계산 완료")
    else:
        # 순차 처리 (작은 파일 또는 워커 수가 1인 경우)
        chunk_counts = [
            len(chunk_sequence(seq, max_len=max_len, stride=stride))
            for seq in sequences
        ]
    
    total_chunks = sum(chunk_counts)

    # 파일명에 시퀀스 개수 정보 추가
    def _append_suffix(path: str, suffix: str) -> str:
        base, ext = os.path.splitext(path)
        if not ext:
            ext = ".h5"
        return f"{base}{suffix}{ext}"

    save_path_with_count = _append_suffix(save_path, f"_{num_sequences}sq")
    if save_path_with_count != save_path:
        print(f"💾 저장 경로 변경: {save_path} -> {save_path_with_count}")
    save_path = save_path_with_count

    print(f"📊 통계:")
    print(f"   - 시퀀스 수: {num_sequences}")
    print(f"   - 총 청크 수: {total_chunks}")
    print(f"   - 평균 청크/시퀀스: {total_chunks / num_sequences:.2f}")

    with h5py.File(save_path, "w") as f:
        # 청크 임베딩 저장 공간
        emb_dataset = f.create_dataset(
            "embeddings",
            shape=(total_chunks, EMBEDDING_DIM),
            dtype=np.float16,
            chunks=(min(100, total_chunks), EMBEDDING_DIM),
            compression="gzip",
            compression_opts=4,
        )

        chunk_offsets = np.zeros(num_sequences, dtype=np.int64)
        chunk_counts_arr = np.zeros(num_sequences, dtype=np.int32)
        seq_lengths = np.zeros(num_sequences, dtype=np.int32)
        ids_list = [""] * num_sequences

        current_offset = 0

        pbar = tqdm(
            zip(ids, sequences), 
            total=num_sequences, 
            desc="🔄 청크 임베딩 추출",
            unit="seq",
            ncols=100,
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"
        )
        
        for idx, (protein_id, seq) in enumerate(pbar):
            seq_len = len(seq)
            n_chunks = chunk_counts[idx]
            pbar.set_postfix({"ID": protein_id[:15], "len": seq_len, "chunks": n_chunks})
            
            # 청크 임베딩 추출
            embeddings = embed_chunks(
                seq,
                tokenizer,
                model,
                device,
                max_len=max_len,
                stride=stride,
                window_batch_size=window_batch_size,
            )
            
            # HDF5에 저장
            emb_dataset[current_offset:current_offset + n_chunks] = embeddings.numpy().astype(np.float16)
            
            chunk_offsets[idx] = current_offset
            chunk_counts_arr[idx] = n_chunks
            seq_lengths[idx] = seq_len
            ids_list[idx] = protein_id
            current_offset += n_chunks
        
        pbar.close()

        # 메타데이터 저장
        f.create_dataset("chunk_offsets", data=chunk_offsets)
        f.create_dataset("chunk_counts", data=chunk_counts_arr)
        f.create_dataset("seq_lengths", data=seq_lengths)

        dt = h5py.special_dtype(vlen=str)
        ids_dataset = f.create_dataset("ids", (num_sequences,), dtype=dt)
        ids_dataset[:] = ids_list

        f.attrs["num_sequences"] = num_sequences
        f.attrs["total_chunks"] = total_chunks
        f.attrs["embedding_dim"] = EMBEDDING_DIM
        f.attrs["dtype"] = "float16"
        f.attrs["model"] = model_name
        f.attrs["max_len"] = max_len
        f.attrs["stride"] = stride

    print(f"✅ 저장 완료: {save_path}")
    return save_path


def load_chunk_embeddings_by_index(h5_path: str, idx: int) -> Tuple[np.ndarray, str, int]:
    """
    HDF5에서 특정 인덱스의 청크 임베딩들을 로드.
    
    Returns:
        embeddings: (num_chunks, EMBEDDING_DIM) 형태의 임베딩
        protein_id: 단백질 ID
        seq_length: 원본 시퀀스 길이
    """
    with h5py.File(h5_path, "r") as f:
        offset = f["chunk_offsets"][idx]
        count = f["chunk_counts"][idx]
        embeddings = f["embeddings"][offset:offset + count]
        protein_id = f["ids"][idx]
        seq_length = f["seq_lengths"][idx]
    return embeddings, protein_id, seq_length


def get_chunk_info(seq_len: int, max_len: int = MAX_LEN, stride: int = STRIDE) -> int:
    """주어진 시퀀스 길이에 대한 청크 개수 반환."""
    if seq_len <= max_len:
        return 1
    
    # 청크 개수 계산
    n_chunks = 1
    start = 0
    while True:
        end = min(start + max_len, seq_len)
        if end == seq_len:
            break
        start += stride
        if start + max_len >= seq_len:
            start = seq_len - max_len
        n_chunks += 1
        if start + max_len >= seq_len:
            break
    
    return n_chunks


__all__ = [
    "MODEL_NAME",
    "EMBEDDING_DIM",
    "MAX_LEN",
    "STRIDE",
    "WINDOW_BATCH_SIZE",
    "extract_chunk_embeddings",
    "load_chunk_embeddings_by_index",
    "chunk_sequence",
    "embed_chunks",
    "get_chunk_info",
]

