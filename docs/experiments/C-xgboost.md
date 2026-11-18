# C · XGBoost 실험 결과 (C-2 ~ C-3)

> 담당: 양서린(C) · 상태: **val 실험 완료 / test 예측 저장(채점은 팀 통합 때 1회)** · 작성: 2026-11-18 · 정리: C/Claude

코드·예측은 `seorin` 브랜치에 있다. 아래 숫자는 모두 **val(질문 22개 × 18모델 = 396개)** 결과다. test 칸은 `미평가`로 둔다([다음 작업 계획](../next-tasks-A-B-C-D.md) 4절 5번).

> **질문 묶음 이용 표시:** 최종 설정의 **질문 중심화**는 같은 `prompt_id` 응답 18개의 평균을 빼므로, 예측할 때 같은 질문의 다른 응답을 함께 쓴다. 계획 4절 9번에 따라 응답 단독 예측(중심화 없음)과 구분해 적는다. 헝가리안 배정(후처리)은 주 지표에 넣지 않고 별도 칸에만 적는다.

## 1. 실험 개요

| 항목 | 내용 |
| --- | --- |
| 담당자 | 양서린(C) |
| OS/CPU/RAM/GPU | 탐색: Linux 6.18 x86_64, CPU 2코어, RAM 7GB, GPU 없음 (Claude 작업 환경) · 최종 재학습: Kaggle Notebook(Linux) |
| Python / NumPy / pandas / scikit-learn 버전 | 탐색: 3.13.18 / 2.5.3 / 3.0.5 / 1.9.1 · Kaggle 최종 재학습: 3.13.16 / 2.1.3 / 2.3.3 / 1.6.1 |
| 추가 사용 라이브러리 버전 | 탐색: xgboost 3.4.1, scipy 1.18.1 · Kaggle: xgboost 3.4.1, scipy 1.16.3 (2026-11-18 확인). 두 환경의 val BA는 같았다(seed 3개 앙상블 0.3813) |
| 실험 모델 | XGBoost (`XGBClassifier`, `multi:softprob`) |
| 모델 구분 | 고전 ML |
| 실험 날짜 | 2026-11-15 ~ 2026-11-16 (탐색), 2026-11-18 (최종 재학습·예측) |
| 실험 환경 | CPU, `tree_method=hist` |
| 코드 | `seorin` [`02715c9`](https://github.com/seoultech-Machine-Learning-26-2/LLM-name-tagging/blob/02715c9/experiments/run_xgb_official.py) `experiments/run_xgb_official.py` (탐색, 실행한 원본 그대로: 평가 코드를 `official_eval.py`(= `eval/evaluate.py` 사본), 데이터를 개인 업로드 경로로 읽음) · 최종 재학습·test 예측은 [`notebooks/kaggle_session_2026-11-18/llmnametagging.ipynb`](https://github.com/seoultech-Machine-Learning-26-2/LLM-name-tagging/blob/e5b3c9b/notebooks/kaggle_session_2026-11-18/llmnametagging.ipynb) 셀 B·C |
| 실행 명령 | `python run_xgb_official.py` (원본은 `official_eval.py`·`features/`와 같은 폴더, 데이터 경로는 실행 환경에 맞게 바꿔야 함). 2026-11-18 경로만 저장소 기준으로 바꿔 다시 실행했을 때 아래 표와 같은 값이 나왔다 |

## 2. 실험 설정

### 2.1 데이터 설정

| 항목 | 내용 |
| --- | --- |
| 데이터셋 | Name that LLM (2026) |
| 전체 데이터 | 2,699개 |
| Train / Validation / Test | 공통 분할 사용 (`data/split/`, 1,889 / 396 / 414) |
| 분할 기준 | `prompt_id` 기준 |
| Random Seed | 42 (분할) · 학습 seed 42/43/44 |
| 분류 대상 | 개발사 6개 / 개별 모델 18개 |

### 2.2 특징 추출 및 전처리

| 항목 | 내용 |
| --- | --- |
| 입력 데이터 | `text` 원문 (그룹 4는 `prompt_id`도 사용) |
| 전처리 방식 | 변환기는 모두 train에만 fit. 그룹 4: 같은 `prompt_id` 18개 응답의 특징 평균을 빼는 질문 중심화 |
| 특징 추출 방식 | B `features/stylometry.py` 문체 32개 + 기능어 비율 99개(단어 수로 나눔) + 문자 3~5그램 TF-IDF(`min_df=2`, 30,000개, `sublinear_tf`, 대소문자 유지; B `build.py` 설정) → `TruncatedSVD(100)` |
| 주요 특징 | 문자 n-gram SVD 성분, 기능어 비율, 대시·문장부호 빈도, 평균 단어 길이 (4.2절) |
| 특징 차원 수 | 32 → 131 / 132 → 231 |

### 2.3 모델 및 하이퍼파라미터

| **항목** | **기록** |
| --- | --- |
| 설정과 탐색 범위 | 그룹 1: `max_depth` 3/6 × `learning_rate` 0.05/0.1 · 그룹 2: 깊이 3 + 규제(`min_child_weight=5`, `reg_lambda=10`) · 그룹 3: 특징 추가 3종 · 그룹 4: 질문 중심화 |
| 고정한 조건 / 바꾼 조건 | 고정: `n_estimators=3000` + val `mlogloss` 조기 종료 50회, `subsample=0.8`, `colsample_bytree=0.8`, `tree_method=hist` · 그룹마다 한 가지만 바꿈 |
| 학습 seed | 그룹 1·3(단일 특징 추가): 42 · 그룹 2·3(전체)·4: 42/43/44 |
| 최종 설정 / 선택 이유 | 깊이 3, 학습률 0.05, 규제, 특징 231개, 질문 중심화. val BA +1%p 이상일 때만 채택 규칙으로 단계마다 채택. 깊이 6이 val은 약간 높았지만(0.227~0.235) train BA 0.94~0.96으로 과적합이 커서, 이후 실험은 깊이 3 + 규제로 고정 |
| 확률 출력 / 보정 방식 | `predict_proba` 그대로 사용(보정 없음). 최종은 seed 3개 확률 평균 |

## 3. 모델 학습 결과

최종 설정(full + 질문 중심화)의 seed 3개 확률 평균 예측(`preds/val_C_XGB.csv`)을 공통 `eval/evaluate.py`로 계산했다.

### 3.1 개발사 분류 (6 Classes)

| 평가 지표 | Validation | Test |
| --- | --- | --- |
| Accuracy | 0.7500 | 미평가 |
| Precision (Macro) | 0.6658 | 미평가 |
| Recall (Macro) = `company_BA` (확률 합산) | 0.6000 | 미평가 |
| F1-score (Macro) | 0.6123 | 미평가 |
| F1-score (Weighted) | 0.7330 | 미평가 |
| Log loss / ECE | 0.7165 / 0.0952 | 미평가 |

`company_BA_argmax`(1위 모델의 개발사) = 0.6114.

### 3.2 개별 모델 분류 (18 Classes)

| 평가 지표 | Validation | Test |
| --- | --- | --- |
| Accuracy (= BA, 클래스당 22개로 같음) | 0.3813 | 미평가 |
| Precision (Macro) | 0.3872 | 미평가 |
| Recall (Macro) | 0.3813 | 미평가 |
| F1-score (Macro) | 0.3801 | 미평가 |
| F1-score (Weighted) | 0.3801 | 미평가 |
| Log loss / ECE | 1.7618 / 0.0703 | 미평가 |
| 형제 정확도 | 0.5078 | 미평가 |

### 3.3 탐색 결과 (val, 18개 모델 BA)

| 그룹 | 설정 | 특징 수 | BA (seed 평균 ± 표준편차) | 배정 BA (후처리) | train BA | 트리 수 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 문체 32, 깊이 3, lr 0.05 | 32 | 0.202 | 0.265 | 0.740 | 175 |
| 1 | 깊이 3, lr 0.1 | 32 | 0.210 | 0.265 | 0.757 | 92 |
| 1 | 깊이 6, lr 0.05 | 32 | 0.227 | 0.237 | 0.963 | 82 |
| 1 | 깊이 6, lr 0.1 | 32 | 0.235 | 0.245 | 0.941 | 34 |
| 2 | 깊이 3 + 규제 | 32 | 0.227 ± 0.004 | 0.272 | 0.644 | 224 |
| 3 | + 기능어 | 131 | 0.242 | 0.298 | 0.938 | 383 |
| 3 | + 문자 SVD | 132 | 0.237 | 0.290 | 0.989 | 299 |
| 3 | + 기능어 + 문자 SVD (**응답 단독 최종**) | 231 | **0.258 ± 0.001** | 0.324 | 0.999 | 398 |
| 4 | + 질문 중심화 (**최종, 질문 묶음 이용**) | 231 | **0.380 ± 0.001** | 0.423 | 1.000 | 538 |

- seed 3개 확률 평균으로 앙상블하면 0.3813(3.2절). seed별 평균(0.380)과 구분한다.
- 표준편차는 seed 사이 값이며, 신뢰구간은 아니다.
- train 내부 질문 단위 5-fold(GroupKFold)에서도 중심화 효과를 확인했다: 중심화 없음 0.294 ± 0.025 → 중심화 0.344 ± 0.010 (배정 포함 0.324 → 0.383). 이 교차검증은 트리 300개로 고정하고, fold마다 TF-IDF·SVD를 그 fold의 train 부분에서 다시 학습했다([`experiments/cv.py`](https://github.com/seoultech-Machine-Learning-26-2/LLM-name-tagging/blob/02715c9/experiments/cv.py)). 문체 32개는 B 공통 코드가 아니라 같은 정의로 따로 구현한 [`experiments/bfeat.py`](https://github.com/seoultech-Machine-Learning-26-2/LLM-name-tagging/blob/02715c9/experiments/bfeat.py)로 계산했다.

### 3.4 학습 정보

| 항목 | 결과 |
| --- | --- |
| 학습 시간 | 최종 설정 seed 3개 합계 약 89초 (CPU 2코어) |
| 최종 선택된 모델 | XGBoost, 특징 231개 + 질문 중심화, seed 42/43/44 확률 평균 |
| 최종 Epoch (해당 시) | 부스팅 라운드 627 / 621 / 538 (seed 42/43/44, 조기 종료) |
| 주요 특이사항 | val을 조기 종료에 사용하므로 val 점수가 약간 낙관적일 수 있다. 최종 모델은 train만으로 학습했다(train+val 재학습 없음) |

## 4. 모델 분석

### 4.1 Confusion Matrix

**개발사 6개 분류** (행=실제, 열=예측, 확률 합산 기준)

| 실제 \ 예측 | Anthropic | Google | Moonshot | OpenAI | Qwen | ZAI |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Anthropic (110) | **93** | 0 | 1 | 2 | 1 | 13 |
| Google (88) | 3 | **84** | 0 | 0 | 1 | 0 |
| Moonshot (22) | 6 | 0 | **4** | 10 | 0 | 2 |
| OpenAI (110) | 6 | 0 | 2 | **93** | 5 | 4 |
| Qwen (22) | 2 | 2 | 0 | 7 | **11** | 0 |
| ZAI (44) | 27 | 0 | 0 | 4 | 1 | **12** |

- 주요 오분류: **ZAI → Anthropic 27/44**, Anthropic → ZAI 13. Moonshot → OpenAI 10/22. 모델 수가 적은 개발사(Moonshot·Qwen)는 큰 개발사로 끌려가는 경향이 있다.

**개별 모델 18개 분류**

- 재현율이 낮은 모델: gpt-5.6-sol 0.091, claude-fable-5 · glm-5.3 · glm-5.3-flash 0.182, gpt-5.6-luna 0.273. 높은 모델: claude-opus-5 · gemini-3.1-pro-preview · qwen3.8-max 0.636.
- 주요 오분류 모델 쌍(개수/22): gemini-3.7-flash → gemini-3.8-flash 10, gemini-3.8-flash → gemini-3.7-flash 8, claude-fable-5 → glm-5.3-flash 7, gpt-5.6-terra → gpt-5.6-sol 6, glm-5.3 → claude-fable-5 6.
- 그래프 파일은 만들지 않았다. 위 수치는 `preds/val_C_XGB.csv`와 `eval/evaluate.py`로 다시 계산할 수 있다.

### 4.2 주요 특징 분석

| 순위 | 특징 | 중요도(gain 비율, seed 3개 평균) | 해석 |
| --- | --- | --- | --- |
| 1 | `svd9` | 0.0247 | 문자 3~5그램 조합의 한 축. 특정 철자·구두점 패턴 묶음 |
| 2 | `svd2` | 0.0209 | 〃 |
| 3 | `em_dash_per_100_chars` | 0.0188 | 대시(—) 사용 빈도. 개발사별 습관 차이가 큼(B EDA와 일치) |
| 4 | `mean_word_length` | 0.0157 | 어휘 선택(긴 단어 선호) |
| 5 | `punctuation_per_100_chars` | 0.0115 | 문장부호 밀도 |
- **분석 방법:** XGBoost `gain` 중요도(질문 중심화 후 특징 기준). 묶음별 합: 문자 SVD 0.496, 기능어 0.339, 문체 32개 0.166.
- **LLM 문체와 관련된 주요 발견:** 단일 특징보다 문자 n-gram과 기능어 같은 분산된 신호가 더 많이 쓰인다. 상관된 특징끼리 중요도가 나뉠 수 있어 중요도만으로 원인을 단정하지 않는다.

### 4.3 오분류 사례 분석

| 실제 모델 | 예측 모델 | 오분류 원인 분석 |
| --- | --- | --- |
| gpt-5.6-terra (`98ea5cffbce0`) | gpt-6-astra (확률 0.90) | 같은 개발사 형제 모델. 문체 통계가 매우 비슷해 개발사는 맞혔지만 세부 모델은 틀림 |
| glm-5.3-flash (`6545c9754511`) | claude-opus-5 (0.89) | ZAI 응답이 Anthropic 쪽 문체 통계와 겹치는 경우. 개발사 혼동행렬의 ZAI → Anthropic 27개와 같은 경향 |
| claude-fable-5 (`6ec853cb8e9f`) | glm-5.3-flash (0.85) | 반대 방향 혼동. 두 모델의 구두점·길이 특징이 비슷하다는 가설이며, 원인은 확정하지 않음 |

## 5. 심화 모델 추가 실험

심화 모델은 [C · Qwen3.5-0.8B + LoRA](C-qwen-lora.md)에 정리했다. XGBoost는 최종 앙상블에서 가중치 0.2로 결합했다.

## 6. 최종 실험 결과 요약

### 6.1 핵심 결과

- **개발사 분류 Test Macro F1:** 미평가 (val 0.6123)
- **개별 모델 분류 Test Macro F1:** 미평가 (val 0.3801)
- **가장 중요한 문체 특징:** 문자 n-gram SVD 성분, 대시 빈도, 평균 단어 길이
- **자주 혼동된 모델:** gemini-3.7-flash ↔ gemini-3.8-flash, claude-fable-5 ↔ glm-5.3 계열, GPT 5.6 형제

### 6.2 결론

**실험을 통해 확인한 점**

- 문체 32개만으로는 0.20~0.24에 머물렀고, 기능어·문자 n-gram을 더하면 응답 단독 예측이 0.258까지 올랐다.
- 질문 중심화가 +12.3%p(0.258 → 0.380)로 가장 큰 개선을 만들었다. 같은 질문의 응답끼리 주제가 같아, 평균을 빼면 주제 정보가 줄고 모델별 차이가 남는다는 가설과 맞는다.

**모델의 장점**

- CPU 몇 분이면 학습되고, 특징 중요도로 어떤 신호를 쓰는지 볼 수 있다.
- 확률 보정이 비교적 좋아(ECE 0.07) 앙상블에서 확률 품질을 보완한다.

**모델의 한계**

- train BA 1.0 대 val 0.38로 과적합이 크다. 규제를 더해도 train을 거의 외운다.
- 질문 중심화는 같은 질문의 응답 18개가 함께 주어질 때만 쓸 수 있다. 응답 하나만 주어지는 상황의 성능은 0.258이다.
- val이 22문항이라 점수 차이 1~2%p는 불확실성이 크다.

---

## 7. 실험 산출물

| 산출물 | 저장 경로 |
| --- | --- |
| 학습 코드 | `seorin`: [`experiments/run_xgb_official.py`](https://github.com/seoultech-Machine-Learning-26-2/LLM-name-tagging/blob/02715c9/experiments/run_xgb_official.py) (탐색), [`cv.py`](https://github.com/seoultech-Machine-Learning-26-2/LLM-name-tagging/blob/02715c9/experiments/cv.py)·[`bfeat.py`](https://github.com/seoultech-Machine-Learning-26-2/LLM-name-tagging/blob/02715c9/experiments/bfeat.py) (5-fold) · 최종 재학습: [`notebooks/kaggle_session_2026-11-18/llmnametagging.ipynb`](https://github.com/seoultech-Machine-Learning-26-2/LLM-name-tagging/blob/e5b3c9b/notebooks/kaggle_session_2026-11-18/llmnametagging.ipynb) 셀 B |
| 모델 체크포인트 | 저장하지 않음. 위 코드로 CPU 약 1.5분에 다시 학습 |
| 설정 / 체크포인트 (필요 시) | 2.3절 |
| 평가 결과 CSV | `seorin`: [`preds/val_C_XGB.csv`](https://github.com/seoultech-Machine-Learning-26-2/LLM-name-tagging/blob/5869bb7/preds/val_C_XGB.csv), [`preds/test_C_XGB.csv`](https://github.com/seoultech-Machine-Learning-26-2/LLM-name-tagging/blob/5869bb7/preds/test_C_XGB.csv) (test 채점 없음) |
| Confusion Matrix | 4.1절 표 (그림 없음) |
| 특징 분석 그래프 | 4.2절 표 (그림 없음) |
| 최종 실험 보고서 | 이 문서 · 요약: [C 실험 요약](C-summary.md) |
