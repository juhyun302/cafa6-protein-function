# 서열 기반 실험 코드

기존 ‘ESM, 1D CNN(지승님)’ 폴더를 역할별 이름으로 정리했습니다. 최종 TA-ESM2 전체 구현을 뜻하지 않는 보존용 실험 코드입니다.

| 파일 | 역할 |
| --- | --- |
| esm2_650m_embeddings.py | ESM2 650M·1280차원 토큰 임베딩 |
| esm2_3b_chunk_embeddings.py | ESM2 3B·2560차원 청크 임베딩 |
| extract_chunk_embeddings.py | 3B 청크 임베딩 생성 진입점 |
| h5_data_loader.py | HDF5 메타데이터와 데이터 로더 |
| hierarchical_cnn.py | CNN·학습 설정·손실 함수 |
| train_cnn.py | CNN 학습 진입점 |
| attention_mil_model.py | Attention MIL 모델 |
| train_attention_mil.py | MIL 학습 진입점 |
| predict_submission.py | MIL 체크포인트의 추론·제출 생성 |
| evaluate.py | 서열 예측의 자체 평가 도구 |
| ancestral_resonance.py | 조상 기반 후처리 실험 |

의존성은 environments/sequence.yml 또는 pyproject.toml의 sequence extra를 참고합니다. GPU·큰 임베딩·원본 데이터가 필요하며 기본 데모에는 사용하지 않습니다.

기존 모듈 import는 파일 이름 변경에 맞춰 갱신했습니다. 실제 GPU 학습·추론은 이번 정리에서 실행하지 않았습니다. 해당 스크립트의 --help, setup_paths, 데이터 폴더 구조를 확인하고 자신의 데이터 snapshot과 맞춰 실행해야 합니다.

650M와 3B 임베딩은 크기·표현이 다르므로 서로 바꿔 사용하지 않습니다. 과거 분석 노트북은 ../../notebooks/sequence-baseline-analysis.ipynb에 있습니다.
