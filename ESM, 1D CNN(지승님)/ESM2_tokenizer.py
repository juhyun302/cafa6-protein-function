"""
ESM2 토큰화 및 임베딩 추출 (Stride 기반 겹침 + 평균화)

- 길이 1022 초과 시 슬라이딩 윈도우로 분할하고 겹치는 구간은 평균.
- HDF5 포맷으로 토큰 임베딩을 저장.
"""

from __future__ import annotations

import os
from typing import List, Tuple

import h5py
import numpy as np
import torch
import torch.nn as nn
from Bio import SeqIO
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

# 모델 및 토크나이저 설정
MODEL_NAME = "facebook/esm2_t33_650M_UR50D"
EMBEDDING_DIM = 1280

# 토큰화/슬라이딩 윈도우 설정
MAX_LEN = 1022  # (1024 - CLS/EOS)
STRIDE = 512    # 50% overlap
WINDOW_BATCH_SIZE = 8


def load_model_and_tokenizer(model_name: str = MODEL_NAME):
    """모델과 토크나이저를 로드."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    model = model.to(device)
    model.eval()
    return tokenizer, model, device


def chunk_sequence(seq: str, max_len: int = MAX_LEN, stride: int = STRIDE) -> List[Tuple[int, str]]:
    """슬라이딩 윈도우로 시퀀스를 분할."""
    if len(seq) <= max_len:
        return [(0, seq)]

    windows: List[Tuple[int, str]] = []
    start = 0
    while True:
        end = min(start + max_len, len(seq))
        windows.append((start, seq[start:end]))
        if end == len(seq):
            break
        start += stride
        if start + max_len >= len(seq):
            start = len(seq) - max_len
    return windows


def embed_sequence_with_stride(
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
    슬라이딩 윈도우로 임베딩을 구하고 겹치는 위치는 평균한다.

    Returns:
        torch.Tensor: (len(sequence), EMBEDDING_DIM) on CPU
    """
    windows = chunk_sequence(sequence, max_len=max_len, stride=stride)
    seq_len = len(sequence)

    sums = torch.zeros((seq_len, EMBEDDING_DIM), device=device, dtype=torch.float32)
    counts = torch.zeros((seq_len, 1), device=device, dtype=torch.float32)

    if use_amp is None:
        use_amp = device.type == "cuda"

    for start_idx in range(0, len(windows), window_batch_size):
        batch = windows[start_idx:start_idx + window_batch_size]
        batch_seqs = [w[1] for w in batch]
        batch_starts = [w[0] for w in batch]
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
            with torch.autocast(device_type=device.type, enabled=use_amp):
                outputs = model(**tokenized)
                hidden = outputs.last_hidden_state  # (batch, L+2, dim)

        for i, (start_pos, chunk_len) in enumerate(zip(batch_starts, chunk_lens)):
            chunk_emb = hidden[i, 1:chunk_len + 1, :]  # drop CLS/EOS
            sums[start_pos:start_pos + chunk_len] += chunk_emb
            counts[start_pos:start_pos + chunk_len] += 1

    counts = counts.clamp_min(1.0)
    averaged = sums / counts
    return averaged.to("cpu")


def extract_embeddings_with_stride(
    fasta_path: str,
    save_path: str,
    max_len: int = MAX_LEN,
    stride: int = STRIDE,
    window_batch_size: int = WINDOW_BATCH_SIZE,
    model_name: str = MODEL_NAME,
    limit: int | None = None,
):
    """FASTA → 슬라이딩 윈도우 임베딩 추출 → HDF5 저장."""
    tokenizer, model, device = load_model_and_tokenizer(model_name)

    ids: List[str] = []
    sequences: List[str] = []
    for i, record in enumerate(SeqIO.parse(fasta_path, "fasta")):
        if limit is not None and i >= limit:
            break
        ids.append(record.id)
        sequences.append(str(record.seq))

    num_sequences = len(sequences)
    total_tokens = sum(len(seq) for seq in sequences)

    # 파일명에 시퀀스 개수 정보를 추가 (예: embeddings_5sq.h5)
    def _append_suffix(path: str, suffix: str) -> str:
        base, ext = os.path.splitext(path)
        if not ext:
            ext = ".h5"
        return f"{base}{suffix}{ext}"

    save_path_with_count = _append_suffix(save_path, f"_{num_sequences}sq")
    if save_path_with_count != save_path:
        print(f"💾 저장 경로 변경: {save_path} -> {save_path_with_count}")
    save_path = save_path_with_count

    with h5py.File(save_path, "w") as f:
        emb_dataset = f.create_dataset(
            "embeddings",
            shape=(total_tokens, EMBEDDING_DIM),
            dtype=np.float16,
            chunks=(min(1000, total_tokens), EMBEDDING_DIM),
            compression="gzip",
            compression_opts=4,
        )

        offsets = np.zeros(num_sequences, dtype=np.int64)
        lengths = np.zeros(num_sequences, dtype=np.int32)
        ids_list = [""] * num_sequences

        current_offset = 0

        # tqdm으로 진행 상황 표시
        pbar = tqdm(
            zip(ids, sequences), 
            total=num_sequences, 
            desc="🔄 임베딩 추출",
            unit="seq",
            ncols=100,
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"
        )
        
        for idx, (protein_id, seq) in enumerate(pbar):
            # 현재 처리 중인 시퀀스 정보 표시
            seq_len = len(seq)
            pbar.set_postfix({"ID": protein_id[:15], "len": seq_len})
            
            embedding = embed_sequence_with_stride(
                seq,
                tokenizer,
                model,
                device,
                max_len=max_len,
                stride=stride,
                window_batch_size=window_batch_size,
            )
            length = embedding.shape[0]

            emb_dataset[current_offset:current_offset + length] = embedding.numpy().astype(np.float16)
            offsets[idx] = current_offset
            lengths[idx] = length
            ids_list[idx] = protein_id
            current_offset += length
        
        pbar.close()

        f.create_dataset("offsets", data=offsets)
        f.create_dataset("lengths", data=lengths)

        dt = h5py.special_dtype(vlen=str)
        ids_dataset = f.create_dataset("ids", (num_sequences,), dtype=dt)
        ids_dataset[:] = ids_list

        f.attrs["num_sequences"] = num_sequences
        f.attrs["total_tokens"] = total_tokens
        f.attrs["embedding_dim"] = EMBEDDING_DIM
        f.attrs["dtype"] = "float16"
        f.attrs["model"] = model_name
        f.attrs["max_len"] = max_len
        f.attrs["stride"] = stride

    return save_path


def load_embedding_by_index(h5_path: str, idx: int):
    """HDF5에서 특정 인덱스의 임베딩을 로드."""
    with h5py.File(h5_path, "r") as f:
        offset = f["offsets"][idx]
        length = f["lengths"][idx]
        embedding = f["embeddings"][offset:offset + length]
        protein_id = f["ids"][idx]
    return embedding, protein_id


__all__ = [
    "MAX_LEN",
    "STRIDE",
    "WINDOW_BATCH_SIZE",
    "extract_embeddings_with_stride",
    "load_embedding_by_index",
    "chunk_sequence",
    "embed_sequence_with_stride",
]

