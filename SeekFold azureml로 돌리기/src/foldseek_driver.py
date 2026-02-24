import argparse
import subprocess
import os
import sys
import multiprocessing
import time
import glob
import math
import zipfile
import shutil
from datetime import timedelta

def run_command(command: str):
    try:
        # bash로 실행(파이프/글롭/환경변수 등 호환성 좋음)
        subprocess.run(command, shell=True, check=True, executable="/bin/bash")
    except subprocess.CalledProcessError as e:
        print(f"\n❌ [CRITICAL ERROR] Command Failed!", flush=True)
        print(f"❌ Command: {command}", flush=True)
        print(f"❌ Exit Code: {e.returncode}", flush=True)
        sys.exit(1)

def ensure_dir(path: str) -> str:
    abs_path = os.path.abspath(path)
    os.makedirs(abs_path, exist_ok=True)
    return abs_path

def safe_rmtree(path: str):
    try:
        if os.path.exists(path):
            shutil.rmtree(path)
    except Exception as e:
        print(f"⚠️ Failed to remove {path}: {e}", flush=True)

def safe_remove_glob(patterns):
    for pat in patterns:
        for f in glob.glob(pat):
            try:
                os.remove(f)
            except:
                pass

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_data", type=str, required=True)
    # Azure ML output 바인딩/로컬 테스트 둘 다 지원 (기본은 ./outputs)
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--batch_size", type=int, default=5000)
    parser.add_argument("--max_seqs", type=int, default=20)
    parser.add_argument("--verbosity", type=int, default=1)
    args = parser.parse_args()

    num_cores = multiprocessing.cpu_count()
    print(f"🔥 Detected CPU Cores: {num_cores}", flush=True)

    # Azure ML에서 가장 안전한 출력 위치
    FINAL_OUTPUT_DIR = ensure_dir(args.output_dir or "outputs")
    print(f"💾 Output dir: {FINAL_OUTPUT_DIR}", flush=True)

    # 작업 디렉토리들
    work_dir = os.path.abspath("working_structures")
    tmp_work_dir = os.path.abspath("tmp_processing")

    safe_rmtree(work_dir)
    ensure_dir(work_dir)

    ensure_dir(tmp_work_dir)

    # ------------------------------------------------------------------------
    # ZIP 압축 해제 (input_data에 zip 있으면 첫 zip 사용)
    # ------------------------------------------------------------------------
    input_data = os.path.abspath(args.input_data)
    print(f"📂 Checking input data in: {input_data}", flush=True)

    zip_files = glob.glob(os.path.join(input_data, "*.zip"))
    target_data_dir = work_dir

    if zip_files:
        zip_path = zip_files[0]
        print(f"📦 Found compressed file: {zip_path}", flush=True)
        print("⏳ Extracting zip to local SSD...", flush=True)
        try:
            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                zip_ref.extractall(work_dir)
            print("✅ Extraction Complete!", flush=True)
        except Exception as e:
            print(f"❌ Extraction Failed: {e}", flush=True)
            sys.exit(1)

        valid_items = [i for i in os.listdir(work_dir) if not i.startswith("__")]
        if len(valid_items) == 1 and os.path.isdir(os.path.join(work_dir, valid_items[0])):
            target_data_dir = os.path.join(work_dir, valid_items[0])
        else:
            target_data_dir = work_dir
    else:
        print("📂 No .zip found. Assuming raw folder input.", flush=True)
        target_data_dir = input_data

    print(f"📂 Target data dir: {os.path.abspath(target_data_dir)}", flush=True)

    # ------------------------------------------------------------------------
    # Foldseek 설치
    # ------------------------------------------------------------------------
    print("⬇️ Checking Foldseek...", flush=True)
    if not os.path.exists("foldseek"):
        run_command("wget https://mmseqs.com/foldseek/foldseek-linux-avx2.tar.gz -q")
        run_command("tar -xzf foldseek-linux-avx2.tar.gz")

    foldseek_bin = os.path.abspath("./foldseek/bin/foldseek")
    run_command(f"chmod +x {foldseek_bin}")
    if not os.path.exists(foldseek_bin):
        print(f"❌ Foldseek binary not found: {foldseek_bin}", flush=True)
        sys.exit(1)

    # ------------------------------------------------------------------------
    # 파일 스캔
    # ------------------------------------------------------------------------
    print(f"📂 Scanning .cif files in {target_data_dir}...", flush=True)
    cif_files = glob.glob(os.path.join(target_data_dir, "**", "*.cif"), recursive=True)

    if not cif_files:
        print("❌ Error: .cif files not found!", flush=True)
        sys.exit(1)

    total_files = len(cif_files)
    print(f"📊 Total files found: {total_files:,}개", flush=True)

    BATCH_SIZE = max(1, int(args.batch_size))
    total_batches = math.ceil(total_files / BATCH_SIZE)

    # ------------------------------------------------------------------------
    # 타겟 DB 생성
    # ------------------------------------------------------------------------
    print("🔨 Building Target DB...", flush=True)
    target_db_path = os.path.join(tmp_work_dir, "targetDB")
    run_command(
        f"{foldseek_bin} createdb {target_data_dir} {target_db_path} "
        f"--threads {num_cores} -v {args.verbosity}"
    )

    print("=" * 60, flush=True)
    print(f"🚀 Start Batch Processing: 총 {total_batches} 단계 (batch_size={BATCH_SIZE})", flush=True)
    print("=" * 60, flush=True)

    start_time = time.time()

    for i in range(total_batches):
        batch_num = i + 1
        batch_start = i * BATCH_SIZE
        batch_end = min((i + 1) * BATCH_SIZE, total_files)
        current_files = cif_files[batch_start:batch_end]

        print(f"\n🔄 [Step {batch_num}/{total_batches}] Processing files {batch_start+1}~{batch_end}...", flush=True)

        # 배치용 심볼릭 링크 폴더
        batch_folder = os.path.join(tmp_work_dir, f"batch_files_{batch_num}")
        safe_rmtree(batch_folder)
        ensure_dir(batch_folder)

        # symlink 생성 (중복 파일명 충돌 방지: 폴더명 prefix 붙이기)
        for idx, src_path in enumerate(current_files):
            base = os.path.basename(src_path)
            # 같은 이름이 여러 개 있을 수 있으니 인덱스 prefix
            dst_path = os.path.join(batch_folder, f"{idx:06d}_{base}")
            try:
                os.symlink(src_path, dst_path)
            except FileExistsError:
                pass
            except OSError as e:
                print(f"❌ Symlink failed: {src_path} -> {dst_path} ({e})", flush=True)
                sys.exit(1)

        q_db = os.path.join(tmp_work_dir, f"queryDB_{batch_num}")
        res_db = os.path.join(tmp_work_dir, f"resultDB_{batch_num}")
        tmp_s = os.path.join(tmp_work_dir, f"tmp_s_{batch_num}")

        # 결과는 무조건 outputs(또는 바인딩된 output_dir)로
        tsv_out = os.path.join(FINAL_OUTPUT_DIR, f"results_batch_{batch_num}.tsv")

        # 1) Query DB
        run_command(
            f"{foldseek_bin} createdb {batch_folder} {q_db} "
            f"--threads {num_cores} -v {args.verbosity}"
        )

        # 2) Search (alignment 포함)
        print("   👉 Searching (with alignment)...", flush=True)
        run_command(
            f"{foldseek_bin} search {q_db} {target_db_path} {res_db} {tmp_s} "
            f"--threads {num_cores} --max-seqs {int(args.max_seqs)} -a -v {args.verbosity}"
        )

        # 3) Convert to TSV
        print("   👉 Converting results...", flush=True)
        fmt = "query,target,evalue,bits,alntmscore,qcov,tcov"
        run_command(
            f"{foldseek_bin} convertalis {q_db} {target_db_path} {res_db} {tsv_out} "
            f"--format-output {fmt} --threads {num_cores} -v {args.verbosity}"
        )

        # 청소
        safe_rmtree(batch_folder)
        safe_remove_glob([f"{q_db}*", f"{res_db}*", f"{tmp_s}*"])

        now = time.time()
        elapsed = now - start_time
        avg_time = elapsed / batch_num
        remain_batches = total_batches - batch_num
        eta_str = str(timedelta(seconds=int(avg_time * remain_batches)))
        percent = (batch_num / total_batches) * 100

        print(f"✅ Batch {batch_num} 완료! ({percent:.1f}%) | ETA: {eta_str}", flush=True)
        print(f"   📌 Saved: {tsv_out}", flush=True)

    # work_dir만 제거 (outputs는 절대 건드리지 않음)
    safe_rmtree(work_dir)
    print("\n🎉 모든 작업 완료!", flush=True)
    print(f"📦 결과 확인: {FINAL_OUTPUT_DIR}", flush=True)

if __name__ == "__main__":
    main()
