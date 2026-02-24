"""
ESM2-3B 청크 임베딩 추출 스크립트 (Azure ML Job용)

사용법:
    python extract_chunk_embeddings.py \
        --dataset_path <데이터셋 루트 경로> \
        --dataset_name <데이터셋 폴더 이름> \
        --fasta_relative_path <FASTA 상대 경로> \
        --output_path <출력 HDF5 파일 경로> \
        [--max_len 1022] \
        [--stride 512] \
        [--window_batch_size 4] \
        [--limit None]

예시:
    python extract_chunk_embeddings.py \
        --dataset_path "${{inputs.dataset}}" \
        --dataset_name "CAFA6_82404sq_Asymmetric Loss" \
        --fasta_relative_path "cafa-6-protein-function-prediction/Train/train_sequences.fasta" \
        --output_path "./outputs/embeddings.h5"
"""

import argparse
import os
import sys
from pathlib import Path

# HF_TOKEN 환경 변수 설정 (Azure ML에서 자동으로 주입됨)
# transformers와 huggingface_hub는 환경 변수를 자동으로 읽습니다
HF_TOKEN = os.environ.get("HF_TOKEN", "")
if HF_TOKEN:
    # HuggingFace Hub가 인식할 수 있도록 환경 변수 설정
    os.environ["HF_TOKEN"] = HF_TOKEN
    os.environ["HUGGING_FACE_HUB_TOKEN"] = HF_TOKEN  # 일부 라이브러리는 이 이름 사용
    # HuggingFace 로그인 (명시적 로그인)
    try:
        from huggingface_hub import login
        login(token=HF_TOKEN, add_to_git_credential=False)
        print("✅ HuggingFace 인증 완료")
    except ImportError:
        print("⚠️ huggingface_hub가 설치되지 않았습니다. 환경 변수만 사용합니다.")
    except Exception as e:
        print(f"⚠️ HuggingFace 로그인 실패: {e}. 환경 변수만 사용합니다.")
else:
    print("⚠️ HF_TOKEN이 설정되지 않았습니다. 공개 모델만 다운로드 가능합니다.")

from ESM2_chunk_tokenizer import extract_chunk_embeddings, MODEL_NAME, EMBEDDING_DIM


def main():
    parser = argparse.ArgumentParser(
        description="ESM2-3B 청크 기반 임베딩 추출",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument(
        "--dataset_path",
        type=str,
        required=True,
        help="데이터셋 루트 경로 (Azure ML inputs.dataset)",
    )
    
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help="데이터셋 내부 폴더 이름 (예: 'CAFA6_82404sq_Asymmetric Loss')",
    )
    
    parser.add_argument(
        "--fasta_relative_path",
        type=str,
        default="cafa-6-protein-function-prediction/Train/train_sequences.fasta",
        help="FASTA 파일의 상대 경로 (데이터셋 루트 또는 dataset_name 기준)",
    )
    
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="출력 HDF5 파일 경로",
    )
    
    parser.add_argument(
        "--max_len",
        type=int,
        default=1022,
        help="최대 시퀀스 길이 (기본값: 1022)",
    )
    
    parser.add_argument(
        "--stride",
        type=int,
        default=512,
        help="슬라이딩 윈도우 stride (기본값: 512)",
    )
    
    parser.add_argument(
        "--window_batch_size",
        type=int,
        default=4,
        help="청크 배치 크기 (기본값: 4)",
    )
    
    parser.add_argument(
        "--model_name",
        type=str,
        default=MODEL_NAME,
        help=f"모델 이름 (기본값: {MODEL_NAME})",
    )
    
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="처리할 시퀀스 개수 제한 (테스트용, 기본값: None)",
    )
    
    parser.add_argument(
        "--use_fp16",
        action="store_true",
        default=True,
        help="FP16 정밀도 사용 (메모리 최적화, 기본값: True)",
    )
    
    parser.add_argument(
        "--no_fp16",
        dest="use_fp16",
        action="store_false",
        help="FP16 비활성화 (FP32 사용)",
    )
    
    parser.add_argument(
        "--num_workers",
        type=int,
        default=8,
        help="멀티프로세싱 워커 수 (기본값: 8)",
    )
    
    args = parser.parse_args()
    
    # 경로 구성 (validation.py와 동일한 로직)
    data_path = Path(args.dataset_path)
    
    print("=" * 70)
    print("📂 데이터셋 경로 검증")
    print("=" * 70)
    print(f"🔍 dataset_path: {data_path}")
    print(f"🔍 dataset_name: {args.dataset_name}")
    print(f"🔍 fasta_relative_path: {args.fasta_relative_path}")
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
    
    # 경로 구성 전략: validation.py와 동일
    # 1. data_path에 예상 폴더가 있는지 확인
    expected_folders = ['embedded_fasta_for_1dcnn', 'go_basic', 'cafa-6-protein-function-prediction']
    has_expected_structure = any((data_path / folder).exists() for folder in expected_folders)
    
    if has_expected_structure:
        # data_path가 이미 데이터셋 루트
        dataset_root = data_path
        print(f"✅ data_path가 데이터셋 루트입니다.")
    elif args.dataset_name:
        # data_path/dataset_name이 데이터셋 루트
        dataset_root = data_path / args.dataset_name
        print(f"✅ dataset_name을 사용하여 경로 구성: {dataset_root}")
    else:
        # dataset_name이 없으면 data_path를 그대로 사용
        dataset_root = data_path
        print(f"⚠️ dataset_name이 없어 data_path를 그대로 사용합니다.")
    
    # FASTA 파일 경로 구성
    fasta_path = dataset_root / args.fasta_relative_path
    
    print(f"\n📄 최종 FASTA 경로: {fasta_path}")
    print(f"   존재 여부: {fasta_path.exists()}")
    
    # 입력 파일 확인
    if not fasta_path.exists():
        print(f"\n❌ 오류: FASTA 파일을 찾을 수 없습니다: {fasta_path}")
        print(f"\n🔍 디버깅 정보:")
        print(f"   dataset_root: {dataset_root}")
        print(f"   dataset_root exists: {dataset_root.exists()}")
        if dataset_root.exists():
            print(f"   dataset_root 내용:")
            try:
                for item in sorted(dataset_root.iterdir())[:10]:
                    item_type = "📁" if item.is_dir() else "📄"
                    print(f"      {item_type} {item.name}")
            except Exception as e:
                print(f"      ⚠️ 폴더 읽기 실패: {e}")
        sys.exit(1)
    
    # 출력 디렉토리 생성
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print("🚀 ESM2-3B 청크 임베딩 추출 시작")
    print("=" * 70)
    print(f"모델: {args.model_name}")
    print(f"임베딩 차원: {EMBEDDING_DIM}")
    print(f"입력 파일: {fasta_path}")
    print(f"출력 파일: {output_path}")
    print(f"최대 길이: {args.max_len}")
    print(f"Stride: {args.stride}")
    print(f"배치 크기: {args.window_batch_size}")
    print(f"FP16 사용: {args.use_fp16} {'(메모리 최적화)' if args.use_fp16 else '(FP32)'}")
    print(f"워커 수: {args.num_workers} (멀티프로세싱)")
    if args.limit:
        print(f"제한: {args.limit}개 시퀀스만 처리")
    print("=" * 70)
    
    try:
        # 임베딩 추출 실행
        saved_path = extract_chunk_embeddings(
            fasta_path=str(fasta_path),
            save_path=str(output_path),
            max_len=args.max_len,
            stride=args.stride,
            window_batch_size=args.window_batch_size,
            model_name=args.model_name,
            limit=args.limit,
            use_fp16=args.use_fp16,
            num_workers=args.num_workers,
        )
        
        print("=" * 70)
        print("✅ 임베딩 추출 완료!")
        print(f"저장 경로: {saved_path}")
        print("=" * 70)
        
    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

