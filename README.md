<!-- ============================================================
     CAFA-6 Protein Function Prediction (2nd Project)
     Focus: AlphaFold 3D + Foldseek structural transfer + Ensemble
     ============================================================ -->

<h1 align="center">CAFA-6 단백질 기능 예측 모델 (2차 프로젝트)</h1>

<p align="center">
  <b>AlphaFold 3D 구조 기반 예측(Foldseek) + 멀티모달 앙상블로 GO term 기능 예측 성능 최적화</b>
</p>

<p align="center">
  <!-- 필요하면 배지 추가 -->
  <!-- <img src="https://img.shields.io/badge/Python-3.10%2B-blue?style=flat-square" />
  <img src="https://img.shields.io/badge/Task-GO%20Term%20Prediction-orange?style=flat-square" />
  <img src="https://img.shields.io/badge/Method-AlphaFold%20%2B%20Foldseek%20%2B%20Ensemble-green?style=flat-square" /> -->
</p>

---

## 📌 프로젝트 개요

본 프로젝트는 **CAFA-6 단백질 기능 예측 과제**를 대상으로,  
단백질의 **3D 구조 정보(AlphaFold)** 를 활용해 **구조 유사도 기반 기능 전이(transfer)** 로 GO term 점수를 산출하고,  
팀 내 다른 모델(서열/텍스트/네트워크 등)과 **앙상블(ensemble)** 하여 최종 예측 성능을 극대화하는 모델을 개발합니다.

---

## 🎯 목표

- AlphaFold로 확보한 **3D 구조**를 기반으로 단백질 기능을 예측
- Foldseek를 이용한 **구조 유사도 검색 + GO term 전이** 파이프라인 구축
- 다양한 정보원 기반 모델 결과를 결합하는 **앙상블 전략**으로 최종 성능 향상

---

## ✅ 우리가 한 일 (핵심 요약)

### 1) 3D 구조 기반 예측 파이프라인 (AlphaFold + Foldseek)
- AlphaFold 구조 데이터(`.cif`) 수집/정리
- Foldseek DB 구성 및 구조 검색 파이프라인 구축
- 타겟 단백질 → 구조적으로 유사한 단백질 검색 → **GO term 전이**
- GO term별 **예측 점수(score) 생성**
- (옵션) GO DAG 전파(ancestor propagation) 등 후처리로 점수 보정 가능

### 2) 멀티모달 앙상블로 최종 모델 구축
- 3D 구조 기반 점수 + 팀원 모델(서열/텍스트/네트워크 등) 점수를 결합
- **가중치/랭킹 기반 앙상블**로 최종 GO 예측 생성
- 내부 검증/평가 지표 기반으로 최적 조합 탐색

---

## 🧠 전체 파이프라인

```text
Protein (Sequence / ID)
   ↓
AlphaFold 3D Structure (CIF)
   ↓
Foldseek Structural Search
   ↓
Nearest Neighbors (structural homologs)
   ↓
GO Term Transfer + Scoring
   ↓
3D-based GO Scores
   ↓
Ensemble (3D + other models)
   ↓
Final GO Prediction
