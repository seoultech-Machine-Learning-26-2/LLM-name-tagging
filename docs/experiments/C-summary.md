# C · 실험 요약: XGBoost + Qwen3.5-0.8B LoRA

> 담당: 양서린(C) · 상태: **val 실험 완료 / test 예측 저장(채점은 팀 통합 때 1회)** · 작성: 2026-11-18 · 수정: 2026-11-18 (Q-10 결정 3·4 반영) · 정리: C/Claude
> 질문: 부스팅과 추가 학습(LoRA)이 비용 대비 어떤 효과를 내는가? · 모든 점수는 val 기준이며, test는 12월 팀 통합 때 1회 채점한다.

상세 결과(결과 템플릿 형식): [C · XGBoost](C-xgboost.md) · [C · Qwen3.5-0.8B + LoRA](C-qwen-lora.md)
코드·예측: `seorin` 브랜치 [`02715c9`](https://github.com/seoultech-Machine-Learning-26-2/LLM-name-tagging/tree/02715c9/experiments)(XGBoost 원본 스크립트), [`5869bb7`](https://github.com/seoultech-Machine-Learning-26-2/LLM-name-tagging/tree/5869bb7/preds)(예측), [`e5b3c9b`](https://github.com/seoultech-Machine-Learning-26-2/LLM-name-tagging/tree/e5b3c9b/notebooks)(Kaggle 노트북 원본: 학습 Version 2, 앙상블·test 예측 세션), [`c5bb47c`](https://github.com/seoultech-Machine-Learning-26-2/LLM-name-tagging/blob/c5bb47c/notebooks/kaggle_session_2026-11-18_xgb_seed/llmnametagging.ipynb)(XGBoost seed별 예측 재실행, Q-10 결정 4)

> **질문 묶음 이용 표시:** ①·②·③의 최종 설정은 **질문 중심화**(같은 `prompt_id` 응답 18개의 평균을 빼기)를 써서, 예측할 때 같은 질문의 다른 응답을 함께 쓴다. 응답 하나만 보는 예측의 최고값은 XGBoost 0.258, Qwen LoRA(Q2) 0.328이다. **배정 BA**는 헝가리안 배정 후처리 결과라 주 지표에 넣지 않는다([다음 작업 계획](../next-tasks-A-B-C-D.md) 4절 9번).

## 0. 한눈에 보기

| 구분 | 최종 설정 | 순수 BA (주 지표) | 배정 BA | 회사 BA | 형제 정확도 | 학습 비용 |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| 기준 | 무작위 선택 | 0.056 | – | – | – | – |
| 기준 | 팀 기준선 (문자 TF-IDF + LR) | 0.290 | 0.404 | 0.452 | 0.402 | CPU 수 초 |
| ① 고전 | XGBoost + 질문 중심화 (3 seed) | 0.381 | 0.432 | 0.600 | 0.508 | CPU 수 분 |
| ② 심화 | Qwen LoRA 6개 앙상블 (V3×3 + Q4×3) | 0.548 | 0.674 | 0.705 | 0.638 | T4 GPU 약 7시간 |
| ③ 최종 | ② × 0.8 + ① × 0.2 | **0.551** | 0.672 | 0.706 | 0.635 | ① + ② |

- 순수 BA: 응답 하나만 보고 작성 모델을 예측한 Balanced Accuracy (주 지표)
- 배정 BA: 같은 질문의 응답 18개에 모델을 하나씩 배정(헝가리안)한 결과 (후처리, 보조 지표, 별도 기록)
- 응답 단독 예측(질문 묶음 미사용): XGBoost 0.258 ± 0.001, Qwen LoRA(Q2) 0.328
- 회사 BA: 18개 확률을 개발사 6개로 합산한 뒤의 BA
- 형제 정확도: 1위 모델의 개발사를 맞힌 응답 중 모델까지 맞힌 비율 (모델 1개인 개발사 제외)

**요약:** 질문 중심화가 고전·심화 모두에서 가장 큰 개선을 만들었다. 최종 앙상블은 기준선 대비 +26%p 높지만, 같은 개발사 형제 모델 구분은 여전히 어렵다.

## 1. 고전 모델: XGBoost

### 입력 특징 (총 231개)

| 묶음 | 개수 | 설명 |
| --- | ---: | --- |
| 문체 특징 | 32 | B의 `stylometry.py` 사용 (train에서 상수 9개) |
| 기능어 빈도 | 99 | 주제와 무관한 단어(the, of, however 등)의 비율 |
| 문자 n-gram | 100 | 문자 3~5그램 TF-IDF(min_df 2, 30,000개, sublinear) → SVD 100차원 (B `build.py` 설정) |
| 질문 중심화 | – | 같은 질문 응답 18개의 평균을 빼 주제 정보 제거 |

설정: `max_depth=3, learning_rate=0.05, min_child_weight=5, reg_lambda=10, subsample=0.8, colsample_bytree=0.8`, val 조기 종료(50). 변환기는 모두 train에만 fit했다.

### 실험 과정 (val BA +1%p 이상일 때만 채택)

| 단계 | 추가한 것 | val BA | 변화 |
| --- | --- | ---: | ---: |
| 1 | 문체 32개 | 0.202 | – |
| 2 | + 규제 (3 seed) | 0.227 | +2.5%p |
| 3 | + 기능어 | 0.242 | +1.5%p |
| 4 | + 문자 n-gram (3 seed) | 0.258 | +1.6%p |
| 5 | + 질문 중심화 (3 seed) | **0.381** | **+12.3%p** |

train 내부 질문 단위 5-fold 교차검증에서도 중심화 효과를 확인했다 (0.294 → 0.344).

효과가 없던 시도: Label spreading(0.07~0.14), Self-training, KMeans 군집. 응답 간 거리가 작성 모델보다 주제에 따라 묶였기 때문으로 본다.

## 2. 심화 모델: Qwen3.5-0.8B + LoRA

### 구조와 학습 설정

| 항목 | 설정 |
| --- | --- |
| 본체 | Qwen3.5-0.8B 텍스트 모델 (24층, hidden 1024, 7.5억 파라미터) |
| LoRA | r 16, α 32, dropout 0.05, 학습 파라미터 4,915,200개 (0.649%) |
| 대상 모듈 | `q_proj, k_proj, v_proj, o_proj, in_proj_qkv, in_proj_z, out_proj` |
| 문장 표현 | 마지막 층 mean pooling → 질문 중심화 `[H ; H − 질문 평균]` (2048차원) |
| 헤드 | 일반형(18개 직접 예측) / 계층형: p(모델) = p(개발사) × p(모델 \| 개발사), 개발사 손실 가중치 0.5 |
| 배치 | 질문 1개의 응답 18개 |
| 최적화 | AdamW (LoRA 2e-4, 헤드 1e-3, weight decay 0.01), warmup 10% + cosine, 3 epoch, clip 1.0 |
| 정밀도 | fp32 가중치 + fp16 autocast, gradient checkpointing, 최대 길이 384 (잘림 0%) |
| 보고 기준 | epoch 3 값 (val 최고 epoch을 고르지 않음) |

### 실험 과정 (seed 42)

| ID | 변경 사항 | 순수 BA | 배정 BA | 회사 BA | train BA | 판단 |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| Q1 | 동결 임베딩 + 헤드 | 0.237 | 0.457 | 0.467 | 1.000 | 출발점 |
| Q2 | + LoRA | 0.328 | 0.455 | 0.554 | 0.866 | |
| Q3 | + 질문 중심화 | 0.482 | 0.604 | 0.682 | 0.984 | 채택 (+15.4%p) |
| V1 | + 규제 강화 (label smoothing 0.1, dropout 0.1, wd 0.05) | 0.472 | 0.535 | 0.635 | 0.993 | 미채택 |
| V2 | + 문장 자르기 증강 | 0.470 | 0.561 | 0.671 | 0.950 | 미채택 |
| V3 | + R-Drop (α 1.0) | 0.492 | 0.606 | 0.686 | 0.985 | 채택 (+1.0%p) |
| Q4 | V3 + 계층형 헤드 | 0.500 | 0.636 | 0.715 | 0.977 | seed 1개 기준 +0.8%p로 보류 |

### seed 3개 검증

| 모델 | seed 42 | seed 43 | seed 44 | 평균 ± 표준편차 |
| --- | ---: | ---: | ---: | ---: |
| V3 | 0.492 | 0.480 | 0.482 | 0.485 ± 0.007 |
| Q4 | 0.500 | 0.492 | 0.518 | **0.503 ± 0.013** |

seed 3개 평균에서는 Q4가 V3보다 +1.9%p 높아 채택 기준을 넘었다. seed 1개로는 결론이 달라질 수 있음을 보여 준다. 같은 seed로 재학습해도 GPU 연산 차이로 결과가 조금 달랐다(V3 seed 44: 0.4848 → 0.4823).

## 3. 최종 조합: 앙상블

조합은 결과를 보기 전에 고정했다.
1. seed 앙상블: 같은 설정 seed 3개의 확률 평균
2. 모델 앙상블: V3 × 3 + Q4 × 3, 6개 확률 평균
3. XGBoost 결합: 가중치 0.8 / 0.9 / 1.0 중 val 순수 BA로 선택 → 0.8 (0.5505 / 0.5455 / 0.5480)

| 조합 | 순수 BA | 배정 BA | 회사 BA | 형제 정확도 | LogLoss | ECE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| V3 단일 (seed 42) | 0.492 | 0.606 | 0.686 | 0.579 | 1.543 | 0.169 |
| V3 × 3 seed | 0.513 | 0.644 | 0.682 | 0.603 | 1.371 | 0.118 |
| Q4 × 3 seed | 0.518 | 0.652 | 0.709 | 0.590 | 1.336 | 0.102 |
| V3×3 + Q4×3 | 0.548 | 0.674 | 0.705 | 0.638 | 1.288 | 0.075 |
| **최종 (+ XGB 0.2)** | **0.551** | 0.672 | 0.706 | 0.635 | **1.260** | **0.059** |

- 단일 모델 대비 최종 앙상블: +5.8%p, 질문 단위 paired bootstrap 95% 신뢰구간 [+3.0, +8.3]%p
- seed 앙상블로 약 +2%p, 구조가 다른 V3·Q4 결합으로 약 +3%p 추가
- XGBoost 결합은 순수 BA +0.25%p로 작지만 LogLoss·ECE(확률 보정)가 가장 좋다
- 배정 BA는 XGB 결합 전(0.674)이 약간 높았으나, 사전 규칙(순수 BA 기준)에 따라 최종 조합을 유지했다

## 4. 분석

1. **질문 중심화가 가장 큰 개선 요인이다.** 같은 질문의 응답은 주제가 같아, 질문 평균을 빼면 주제 정보는 줄고 모델별 문체 차이가 남는다. (고전 +12.3%p, 심화 +15.4%p)
2. **형제 모델 구분이 핵심 난제다.** 개발사 BA는 0.71이지만 모델 BA는 0.55다. 선행 연구(Idiosyncrasies in LLMs)에서도 같은 개발사 모델 구분은 약 60% 수준이다.
3. **과적합이 남아 있다.** train BA 0.98 vs val 0.50. 규제·증강은 효과가 없었고 R-Drop만 소폭 개선했다. 학습 곡선이 데이터 증가에 따라 계속 오르고 있어 데이터 부족이 근본 원인으로 보인다.
4. **비용 대비 효과:** Qwen 앙상블은 XGBoost보다 +17%p 높지만 GPU 학습 비용이 크다. XGBoost는 앙상블에서 확률 보정에 기여한다.

## 5. 파일

| 파일 | 내용 |
| --- | --- |
| `seorin` `preds/val_C_FINAL.csv`, `preds/test_C_FINAL.csv` | 최종 앙상블 예측 (팀 앙상블용) |
| `val_C_XGB.csv`, `test_C_XGB.csv` | XGBoost 3 seed 확률 평균 예측 |
| `val_C_XGB_seed42.csv` ~ `seed44`, `test_C_XGB_seed42.csv` ~ `seed44` | XGBoost seed별 예측 (6개, Q-10 결정 4, `c262b0d`). 세 파일 평균 = `C_XGB` |
| `val_V3_rdrop_seed42~44`, `val_Q4_hier_seed42~44` ↔ `test_C_V3_seed42~44`, `test_C_Q4_seed42~44` | 팀 앙상블에 쓰는 Qwen 개별 파일 6쌍 (Q-10 결정 3) |
| `test_C_Q2.csv` ~ `test_C_Q4_seed44.csv` | Qwen 개별 모델 test 예측 (8개) |
| `test_nolabel.csv` | 정답 열을 제거한 test (팀원 예측용) |

형식은 팀 공통(`id` + `CLASSES` 18개 확률)이며 `evaluate.py`의 `load_run`으로 검사할 수 있다. 저장된 어댑터로 val을 다시 예측해 학습 당시 예측과 일치함(8개 중 7개 100%, 1개 99.7%)을 확인했다.

## 6. 실행 환경

- 학습: Colab T4(seed 42) → Kaggle T4(seed 43·44), 1회 학습 약 1시간 10분, 최대 메모리 5.4GB
- XGBoost 탐색: Linux CPU, Python 3.13.18, scikit-learn 1.9.1, xgboost 3.4.1 (상세는 [C · XGBoost](C-xgboost.md) 1절)
- Kaggle 라이브러리 버전(2026-11-18 확인): Python 3.13.16 / NumPy 2.1.3 / pandas 2.3.3 / scikit-learn 1.6.1, torch 2.11.0+cu130, transformers 5.23.0, peft 0.21.3, accelerate 1.16.0, xgboost 3.4.1, scipy 1.16.3 (flash-linear-attention 버전은 미기록). 학습 당시 버전과 다를 수 있다(매번 최신 설치)
- XGBoost seed별 파일: Kaggle GPU T4 세션(가속기 켬, XGBoost는 CPU로 학습), Python 3.13.18, xgboost 3.4.1, scikit-learn 1.6.1, numpy 2.1.3, pandas 2.3.3, scipy 1.16.3. 가속기를 끈 CPU 전용 세션에서는 조기 종료 트리 수가 달라졌다(660/629, 하드웨어에 따른 부동소수점 차이로 추정)
- 모델 체크포인트: 깃허브 미포함. 저장 위치·복원 방법은 [C · Qwen LoRA](C-qwen-lora.md) 7절
