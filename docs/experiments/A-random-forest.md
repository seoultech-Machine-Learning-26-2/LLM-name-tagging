# A: Random Forest 실험 결과

> 담당: 좌은진(A) · 작성일: 2026-11-17 · 수정일: 2026-11-22 · 정리: A/Claude · 상태: val 완료(응답 단독 3~6절, 질문 중심화 9절), test 예측 저장·채점 보류(팀 일괄 채점 대기)
> 미완료: 데이터 출처(A-1)

## 1. 실험 개요

| 항목 | 내용 |
| --- | --- |
| 담당자 | 좌은진 (A) |
| OS/CPU/RAM/GPU | Kaggle Notebook, Linux-6.18.55+-x86_64-with-glibc2.39, CPU 4코어 (GPU 미사용) |
| Python / NumPy / pandas / scikit-learn | 3.13.18 / 2.1.3 / 2.3.3 / 1.6.1 |
| 추가 라이브러리 | SciPy 1.16.3, matplotlib 3.10.0 |
| 실험 모델 | `RandomForestClassifier` |
| 모델 구분 | 고전 ML |
| 실험 날짜 | 2026-11-18 (2026-11-17 실행과 val 결과 동일) |
| 실험 환경 | Kaggle Notebook (`notebooks/A_rf_kaggle.ipynb`), `eunjin` 브랜치 코드 커밋 `a4a205a` (RF 코드는 `b416867`과 동일) |

## 2. 실험 설정

### 2.1 데이터 설정

| 항목 | 내용 |
| --- | --- |
| 데이터셋 | Name that LLM (2026) — 출처·사용 조건 미기록 |
| 전체 데이터 | 2,699개 |
| Train / Validation / Test | 1,889 / 396 / 414 (A의 공통 분할, 질문 105 / 22 / 23개) |
| 분할 기준 | `prompt_id` 기준, 분할 간 질문 겹침 0 |
| Random Seed | 분할 42, 학습 42 / 43 / 44 |
| 분류 대상 | 개별 모델 18개 (개발사 6개는 확률 합산으로 별도 평가) |

### 2.2 특징 추출 및 전처리

| 항목 | 내용 |
| --- | --- |
| 입력 데이터 | `text` 열 |
| 전처리 방식 | 없음. 트리는 값의 크기에 영향을 받지 않아 스케일링하지 않음 |
| 특징 추출 방식 | B의 공통 문체 특징 (`features/stylometry.py`) |
| 주요 특징 | 길이, 단어·문장, 문자 구성, 문장부호 비율 |
| 특징 차원 수 | 32 (train에서 상수 9개: `line_count`, `blank_line_ratio`, 제목·글머리·번호·코드 경계 비율, 굵은 글씨·링크·백틱 비율 → 실제로 변하는 특징 23개) |

문체 특징은 글 하나만 보고 계산하는 고정 공식이라 데이터에서 학습되는 값이 없다. val/test 정보가 학습에 들어가지 않는다.

### 2.3 모델 및 하이퍼파라미터

| **항목** | **기록** |
| --- | --- |
| 설정과 탐색 범위 | 기본 `n_estimators=100`, `max_depth=None`, `min_samples_leaf=1` (계획 문서의 시작 설정) |
| 고정한 조건 / 바꾼 조건 | 비교 1: 트리 수 50 / 100 / 300 (깊이 None 고정). 비교 2: 최대 깊이 5 / 10 / None (트리 100 고정). 100/None은 공유해 총 5개 설정 |
| 학습 seed | 모든 설정을 42 / 43 / 44로 반복 |
| 최종 설정 / 선택 이유 | **d10** (`n_estimators=100`, `max_depth=10`). val 18개 모델 BA 평균 1위와 차이가 1위의 seed 표준편차 이내인 설정 중 val log loss 최소 (C의 기존 규칙) |
| 확률 출력 / 보정 방식 | `predict_proba` (트리별 잎 비율의 평균), 별도 보정 없음 |

## 3. 모델 학습 결과

최종 설정 d10, seed 3개 평균. **test는 채점 보류**(8절 참고).

### 3.1 개발사 분류 (6 Classes)

| 평가 지표 | Validation | Test |
| --- | --- | --- |
| Accuracy (확률 합산) | 0.5699 | 채점 보류 |
| Precision (Macro) | 0.3964 | 채점 보류 |
| Recall (Macro) = `company_BA` | 0.3750 | 채점 보류 |
| F1-score (Macro) | 0.3289 | 채점 보류 |
| F1-score (Weighted) | 0.4984 | 채점 보류 |
| Log loss / ECE | 1.1640 / 0.0852 | 채점 보류 |
| 참고: `company_BA_argmax` (1위 모델의 개발사) | 0.4407 | 채점 보류 |

### 3.2 개별 모델 분류 (18 Classes)

| 평가 지표 | Validation | Test |
| --- | --- | --- |
| Accuracy | 0.2054 ± 0.0130 | 채점 보류 |
| Precision (Macro) | 0.1792 | 채점 보류 |
| Recall (Macro) = BA | 0.2054 ± 0.0130 | 채점 보류 |
| F1-score (Macro) | 0.1848 | 채점 보류 |
| F1-score (Weighted) | 0.1848 | 채점 보류 |
| Log loss / ECE | 2.4107 / 0.0365 | 채점 보류 |
| 형제 정확도 | 0.3509 | 채점 보류 |

클래스별 표본 수가 같아(val 각 22개) BA와 Accuracy, Macro F1과 Weighted F1이 같다.

### 3.3 조건별 비교 (val, seed 3개 평균 ± 표준편차)

| 설정 | 바꾼 조건 | train BA | val BA | train−val | val Macro F1 | val log loss | 학습 시간(초) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Dummy (사전확률) | 기준선 | – | 0.0556 | – | 0.0058 | – | – |
| n50 | 트리 50 | 1.000 | 0.2020 ± 0.0116 | 0.798 | 0.1885 | 4.628 | 0.30 |
| n100 | 기본 | 1.000 | 0.2096 ± 0.0091 | 0.790 | 0.1948 | 3.557 | 0.54 |
| n300 | 트리 300 | 1.000 | 0.2104 ± 0.0130 | 0.790 | 0.1979 | 2.572 | 1.60 |
| d5 | 깊이 5 | 0.460 | 0.2113 ± 0.0172 | 0.248 | 0.1770 | 2.459 | 0.37 |
| **d10** | 깊이 10 | 0.973 | 0.2054 ± 0.0130 | 0.767 | 0.1848 | **2.411** | 0.48 |

| 참고 기준선 (val, 18개 모델 BA) | 입력 | val BA |
| --- | --- | ---: |
| TF-IDF + 로지스틱 회귀 (C) | 문자 2~6그램 TF-IDF | 0.2904 |
| XGBoost (C), 깊이 3·lr 0.05, seed 42 | 문체 32개 | 0.202 |
| XGBoost (C), 깊이 3 + 규제, seed 3개 | 문체 32개 | 0.227 ± 0.004 |
| XGBoost (C), 깊이 6·lr 0.1, seed 42 | 문체 32개 | 0.235 |

- 다섯 설정의 val BA 차이(최대 0.009)는 seed만 바꿨을 때의 흔들림(0.009~0.017)보다 작다. 트리 수와 깊이는 val 성능을 의미 있게 바꾸지 못했다.
- 선택 규칙상 모든 설정이 후보에 들어가 log loss가 가장 낮은 d10이 선택됐다. val BA 평균 1위는 d5였으나 차이는 seed 변동 범위 안이다.
- RF와 C 기준선은 입력과 알고리즘이 모두 달라, 차이를 RF 알고리즘 탓으로 볼 수 없다.
- **같은 문체 32개 입력의 배깅(RF)과 부스팅(XGBoost) 비교:** RF d10 0.205 ± 0.013, XGBoost 0.202~0.235로 비슷한 범위다([C XGBoost](C-xgboost.md) 3.3절, `seorin` 브랜치). RF는 깊은 트리를 독립적으로 여러 개 만들어 평균하고(분산 감소), XGBoost는 얕은 트리를 순차적으로 더해 앞선 오차를 보완한다(편향 감소). 두 방식 모두 val 0.20~0.24에 머물러, 이 입력에서는 앙상블 방식보다 문체 특징의 정보량이 성능을 제한한다는 5절의 가설과 맞는다. XGBoost는 val을 조기 종료에 써서 val 점수가 약간 낙관적일 수 있고, 특징을 231개로 늘리면 0.258까지 올랐다(C 문서).

### 3.4 학습 정보

| 항목 | 결과 |
| --- | --- |
| 학습 시간 | 설정당 0.30~1.6초 (CPU 4코어, `n_jobs=-1`) |
| 최종 선택된 모델 | d10 (`n_estimators=100`, `max_depth=10`) |
| 최종 Epoch | 해당 없음 |
| 주요 특이사항 | 깊이 제한이 없으면 train BA 1.0 (학습 데이터 암기) |

## 4. 모델 분석

### 4.1 Confusion Matrix

`results/course/A/random_forest/fig_confusion_val.png` (d10 seed 42, val, 행 정규화)

**개발사 6개 분류**
- 재현율(확률 합산): Google 0.886, Anthropic 0.691, OpenAI 0.618, ZAI·Moonshot·Qwen 0
- 모델이 1~2개뿐인 개발사는 합칠 확률이 적어 개발사 예측에서 한 번도 선택되지 않았다. C 기준선에서도 같은 현상이 있었다.

**개별 모델 18개 분류**
- 모델별 재현율: `gpt-5.6-luna`·`gpt-5.6-sol` 0, 가장 높은 `gemini-3.1-pro-preview` 0.455
- 주요 오분류 쌍: `gemini-3.6-flash` → `gemini-3.8-flash` (7개), `gpt-5.6-sol` → `gpt-6-astra`, `gemini-3.8-flash` ↔ `gemini-3.7-flash`, `gemini-3.6-flash` → `gemini-3.1-pro-preview`, `claude-fable-5` → `claude-sonnet-5` (각 6개), `glm-5.3-flash` → `glm-5.3` (5개)
- 오분류 319개 중 42.6%가 같은 개발사 안의 혼동이다. 형제 정확도 0.35는 개발사를 맞힌 다중 모델 개발사 글의 약 65%에서 세부 모델을 틀렸다는 뜻이다.

### 4.2 주요 특징 분석

`results/course/A/random_forest/feature_importance.csv`, `fig_feature_importance.png` (d10, seed 3개 평균)

| 순위 | 특징 | 중요도/계수 등 | 해석 |
| --- | --- | --- | --- |
| 1 | `punctuation_per_100_chars` | MDI 0.081 / 순열 0.024 | 전체 문장부호 비율. B의 EDA에서 Gemini 4개가 낮았다 |
| 2 | `comma_per_100_chars` | MDI 0.069 / 순열 0.008 | 쉼표 비율 |
| 3 | `em_dash_per_100_chars` | MDI 0.069 / 순열 **0.051** | 긴 대시. 순열 중요도 1위. GLM·Claude는 많이, Gemini는 거의 쓰지 않음 |
| 4 | `mean_word_length` | MDI 0.065 / 순열 −0.002 | 평균 단어 길이 |
| 5 | `period_per_100_chars` | MDI 0.062 / 순열 −0.001 | 마침표 비율 |

- **분석 방법:** MDI(train에서 분기할 때 불순도를 줄인 양의 합)와 순열 중요도(val에서 그 특징 값을 섞었을 때 BA가 떨어지는 양)
- **LLM 문체와 관련된 주요 발견:** 두 기준 모두 높은 특징은 긴 대시와 전체 문장부호 비율이다. MDI는 연속형 특징을 높게 평가하는 경향이 있고 train 기준이다. 순열 중요도가 0 근처나 음수인 특징이 많은데, 상관된 다른 특징이 정보를 대신할 수 있기 때문이다. 중요도가 높다고 작성 모델 차이의 원인이라고 단정하지 않는다.

### 4.3 오분류 사례 분석

`results/course/A/random_forest/val_errors_seed42.csv` (d10 seed 42, 확신이 높았던 오류 중 선택)

| 실제 모델 | 예측 모델 | 오분류 원인 분석 |
| --- | --- | --- |
| gemini-3.8-flash (`c60ba8259082`, p010) | gemini-3.1-pro-preview (0.50) | 1,246자로 길고 문장부호 2.0, 긴 대시 0. Google 공통 특징으로 개발사는 맞혔다. train 평균이 가장 짧은 3.8-flash(690자)치고 길어서 더 긴 pro 쪽으로 기운 것으로 보인다 |
| gemini-3.6-flash (`f6a03d6833d5`, p073) | gemini-3.8-flash (0.46) | 문장부호 1.66, 긴 대시 0. Gemini 모델들의 문체 숫자 분포가 크게 겹쳐 형제 모델을 구분할 단서가 부족하다 |
| gpt-5.5 (`fc9a710647c8`, p026) | gemini-3.1-pro-preview (0.42) | gpt-5.5는 train 문장부호 평균이 가장 높은데(3.01) 이 글은 2.21이고 대시가 없어 Gemini 쪽 분포에 들어갔다 |
| claude-sonnet-5 (`79e8a6d3ebeb`, p026) | gemini-3.7-flash (0.40) | 1,501자, 문장부호 1.60, 긴 대시 0. Claude 특유의 긴 대시를 쓰지 않은 글이다. 대시·부호 특징에 의존하므로 평소와 다른 글버릇이면 다른 개발사로 틀린다 |
| glm-5.3 (`e234eecef3b0`, p010) | claude-opus-5 (0.39) | 문장부호 3.47, 긴 대시 0.14. GLM과 Claude는 둘 다 긴 대시를 많이 쓴다(B EDA: glm-5.3 0.239, claude-fable-5 0.227). 같은 글버릇을 공유하는 다른 개발사끼리 혼동한다 |

- 같은 질문(p010, p026)에서 여러 오분류가 나왔다. 질문 형식(서정적 묘사, 찬반 논증 등)에 따라 문장 길이·부호 사용이 달라져 모델 간 차이가 가려질 수 있다는 가설이며, 질문별 정확도 분석은 하지 않았다.
- 가설은 특징 값과 B의 train 평균을 비교한 관찰에 근거한다. 원문을 사람이 판정해 정답을 바꾸지 않았다.

## 5. 과적합과 트리 수 분석

**깊이와 과적합**
- **관찰:** 깊이 제한이 없으면 train BA 1.0. 깊이 5로 제한하면 train−val 차이가 0.79 → 0.25로 줄지만 val BA는 그대로(0.21)였다.
- **가설:** 성능의 한계는 모델 복잡도보다 문체 32개(유효 23개)가 담은 정보량에 있다.
- **근거:** 깊이 비교, 트리 수 비교 모두 val BA가 seed 변동 범위 안에서 같았다.
- **남은 한계:** 같은 RF에 더 많은 정보의 입력(TF-IDF 등)을 넣는 실험은 하지 않았다. 계획상 입력 비교는 B(SVM)와 D(LR)가 담당한다.

**트리 수와 확률**
- 트리 50 → 300에서 val BA는 0.202 → 0.210으로 거의 같았지만 log loss는 4.63 → 2.57로 낮아졌고 학습 시간은 약 5.3배 늘었다.
- 깊이 제한이 없는 트리는 잎마다 한 클래스만 남아 0/1에 가까운 확률을 낸다. 트리가 적으면 정답 확률이 0인 경우가 생겨 log loss가 커진다. 여러 트리의 평균(배깅)이 이를 완화한다.

## 6. 최종 실험 결과 요약

### 6.1 핵심 결과

- **개발사 분류 Validation Macro F1:** 0.3289 (test 채점 보류)
- **개별 모델 분류 Validation Macro F1:** 0.1848 (val BA 0.2054, test 채점 보류)
- **가장 중요한 문체 특징:** 긴 대시(`em_dash_per_100_chars`), 전체 문장부호 비율
- **자주 혼동된 모델:** 같은 개발사 형제 모델 (Gemini flash 계열끼리, `gpt-5.6-sol` → `gpt-6-astra`, `claude-fable-5` → `claude-sonnet-5`)

### 6.2 결론

**실험을 통해 확인한 점**
- 문체 32개만으로 18개 작성 모델을 우연(1/18)의 약 3.7배 수준으로 구분했다 (val BA 0.205).
- 트리 수와 깊이는 val 성능을 의미 있게 바꾸지 못했다. 깊이 제한은 과적합을 줄였지만 일반화 성능은 그대로였다.
- 오류의 상당수는 같은 개발사 모델 사이에서 생겼고, 모델이 1~2개인 개발사는 개발사 단위에서도 예측되지 않았다.

**모델의 장점**
- 학습이 1초 안팎으로 빠르고, 특징 중요도로 어떤 문체를 썼는지 해석할 수 있다.
- 스케일링이 필요 없고 설정에 둔감해 튜닝 부담이 작다.

**모델의 한계**
- 유효 특징이 23개로 적어 문자 TF-IDF 기준선(val 0.290)보다 낮다.
- val 질문이 22개라 점수 불확실성이 크다.

---

## 7. 실험 산출물

아래 파일은 2026-11-18 Kaggle 실행의 원본 출력이다(`eunjin` 브랜치). test 채점 파일은 없다.

| 산출물 | 저장 경로 |
| --- | --- |
| 학습 코드 | `experiments/A_random_forest.py` (`eunjin` 브랜치) |
| Kaggle 노트북 | `notebooks/A_rf_kaggle.ipynb` (`eunjin` 브랜치, 11/18 실행에 사용) |
| 모델 체크포인트 | 저장하지 않음 (재실행 약 30초) |
| 평가 결과 CSV | `results/course/A/random_forest/val_results.csv`, `val_summary.csv` |
| 추가 분석 CSV | `feature_importance.csv`, `val_errors_seed42.csv` |
| Confusion Matrix | `results/course/A/random_forest/fig_confusion_val.png` |
| 특징 분석 그래프 | `results/course/A/random_forest/fig_feature_importance.png`, `fig_config_comparison.png` |
| 예측 파일 | `preds/val_A_rf_{n50,n100,n300,d5,d10}_seed{42,43,44}.csv` |
| 실행 환경 기록 | `results/course/A/random_forest/metadata.json` |

## 8. test 처리 기록과 남은 작업

**test 처리 기록**
- 2026-11-17, 당시 계획(각자 고정한 모델의 test를 1회 평가)에 따라 d10의 test를 `--final d10`으로 1회 채점했다. 설정은 그 전에 val로 고정했고, 채점 뒤 바꾼 설정은 없다.
- 2026-11-18 팀 회의에서 test는 각자 채점하지 않고 팀 앙상블 후 일괄 채점하기로 했다. 그래서 이 채점 결과는 공식 결과로 쓰지 않으며, 이 문서와 브랜치에서 뺐다. 당시 결과 파일은 커밋 `0827bba`에 남아 있다.
- 최종 설정 d10은 바꾸지 않는다.

**남은 작업**
- test 예측 파일: 2026-11-22 질문 중심화 실행(9절)에서 d10의 응답 단독(`raw`)·질문 중심화(`sub`, `concat`) 예측을 seed별로 저장했다(정답 열을 읽지 않고 채점 없음). 응답 단독 `raw`는 3~6절 d10과 같은 모델이다(val 예측이 기존 파일과 동일).
- 데이터 출처·사용 조건 기록 (A-1)

## 9. 질문 중심화 (2026-11-22, 질문 묶음 이용)

2026-11-18 회의 결정(질문함 Q-10 결정 2: 4명 모두 질문 중심화 적용)에 따른 추가 실험이다. 처음에는 심화 모델(DeBERTa)에만 적용했고, 다른 팀원이 모두 고전 모델에도 적용한 것을 확인한 뒤 11/22에 RF에도 적용했다(마감 11/20 이후). 적용 규칙은 질문함 Q-11을 따른다.

### 9.1 방법

- **모델 설정 고정:** 3~6절에서 val로 고른 d10(트리 100, 깊이 10, `min_samples_leaf=1`)을 그대로 쓰고 **입력만** 바꿨다. seed 42/43/44.
- **입력 3종**

| 입력 | 내용 | 특징 수 | 질문 묶음 |
| --- | --- | ---: | --- |
| `raw` | 문체 32개 그대로 (3~6절과 같은 입력) | 32 | 응답 단독 |
| `sub` | X − 같은 `prompt_id` 응답들의 X 평균 (C XGBoost 방식) | 32 | 질문 묶음 이용 |
| `concat` | [X ; X − 질문 평균] (C Qwen·B·D 방식) | 64 | 질문 묶음 이용 |

- 질문 평균은 각 분할(train·val·test) 안에서 같은 질문의 응답끼리만 계산하고 정답(`model`)은 쓰지 않는다.
- `sub`·`concat` 중 선택: val BA 평균 1위와의 차이가 1위의 seed 표준편차 이내면 val log loss가 낮은 쪽(3절과 같은 규칙).
- test는 정답 열을 읽지 않고(`id`·`prompt_id`·`text`만, C `test_nolabel.csv`와 같은 내용) 예측 파일만 저장했다. 채점하지 않았다.
- 실행: Kaggle CPU 4코어, `eunjin` 코드 커밋 `fa048f0`, `notebooks/A_rf_centered_kaggle.ipynb`. 환경은 1절과 같다.

### 9.2 결과 (val, seed 3개)

| 지표 | `raw` (응답 단독) | `sub` (선택) | `concat` |
| --- | ---: | ---: | ---: |
| **18개 모델 BA** (seed 평균 ± 표준편차) | 0.2054 ± 0.0130 | **0.2736 ± 0.0204** | 0.2744 ± 0.0125 |
| 18개 모델 BA (seed 3개 확률 평균 예측) | 0.2247 | 0.2652 | 0.2854 |
| Macro F1 | 0.1848 | 0.2578 | 0.2549 |
| 개발사 BA (확률 합산) | 0.3750 | 0.4112 | 0.4111 |
| 형제 정확도 | 0.3509 | 0.4402 | 0.4438 |
| Log loss / ECE | 2.411 / 0.037 | 2.158 / 0.053 | 2.159 / 0.059 |
| train BA (과적합 정도) | 0.973 | 0.983 | 0.990 |

- `sub`와 `concat`의 val BA 차이는 seed 변동 범위 안이라, 규칙대로 log loss가 조금 더 낮은 `sub`를 골랐다.
- `raw`는 3~6절 d10과 같은 모델이며 val 예측 파일이 기존 파일과 같았다(재현 확인).

### 9.3 중심화 효과

| 집계 기준 | `raw` → `sub` | 비고 |
| --- | --- | --- |
| seed별 BA의 평균 | 0.2054 → 0.2736, **+6.8%p** | 발표 그래프 기준(다른 모델의 seed 평균 표기와 같은 방식) |
| seed 3개 확률 평균 예측, 질문 단위 paired bootstrap 1,000회 | 0.2247 → 0.2652, **+4.1%p, 95% CI [0.3, 7.8]%p** | 신뢰구간이 0보다 큼 |
| 개발사 BA (paired bootstrap) | +3.2%p, 95% CI [1.0, 5.5]%p | |

두 집계 기준의 숫자가 다르다(seed별 BA 평균 vs seed 확률을 평균한 하나의 예측). 같은 질문의 공통 성분을 빼자 문체 32개만으로도 18개 모델 BA가 올랐다. 3~6절에서 트리 수·깊이를 바꿔도 val이 0.21 근처에 머물렀던 것과 대비된다. 문체 숫자에 질문(주제·형식)의 영향이 섞여 있었고, 이를 빼면 작성 모델의 차이가 더 드러난다는 해석과 맞다(5절 가설 보완).

### 9.4 분석 (`sub`, val)

- **특징 중요도:** 긴 대시(`em_dash_per_100_chars`)가 MDI·순열 중요도 모두 1위(순열 0.077)이고, 평균 단어 길이가 순열 중요도 2위(0.031)로 올라왔다(중심화 전 −0.002). 질문 평균을 빼면 같은 질문 안에서 "다른 모델보다 긴 단어를 쓰는지"가 단서가 된 것으로 본다.
- **개발사 재현율 (seed 42):** Google 0.886, Anthropic 0.736, OpenAI 0.709, ZAI 0.068, Moonshot·Qwen 0. 모델 수가 적은 개발사가 예측되지 않는 문제는 남았다(DeBERTa 중심화는 Moonshot 0.409, ZAI 0.364).
- **오분류:** 289개 중 40.1%가 같은 개발사 안의 혼동. 주요 쌍: `gemini-3.8-flash` → `gemini-3.7-flash` 10, `gemini-3.7-flash` → `gemini-3.8-flash` 8, `gemini-3.6-flash` → `gemini-3.1-pro-preview` 7, `claude-opus-4-8` → `claude-sonnet-5` 7.

### 9.5 한계

- 같은 질문의 응답 묶음이 있어야 쓸 수 있다(계획 4절 9번: 응답 단독 결과와 구분해 적는다).
- 모델 설정(d10)은 중심화 전 입력으로 고른 것이다. 중심화 입력에 맞춰 다시 고르지 않았다(입력만 바꾼 통제 비교).
- 마감(11/20) 이후 실행했다.

### 9.6 산출물 (`eunjin` 브랜치)

| 산출물 | 저장 경로 |
| --- | --- |
| 코드 | `experiments/A_rf_centered.py` (`A_random_forest.py`의 함수 사용, 기존 파일은 수정하지 않음) |
| Kaggle 노트북 | `notebooks/A_rf_centered_kaggle.ipynb` (이 실행에 사용) |
| 결과 CSV | `results/course/A/random_forest_centered/val_results.csv`, `val_summary.csv`, `val_compare_raw_vs_sub.csv`, `feature_importance.csv`, `val_errors_seed42.csv` |
| 그림 | `results/course/A/random_forest_centered/fig_input_comparison.png`, `fig_feature_importance.png`, `fig_confusion_val.png` |
| 실행 환경 기록 | `results/course/A/random_forest_centered/metadata.json` |
| 예측 파일 | `preds/val_A_rf_d10_{raw,sub,concat}_seed{42,43,44}.csv`, `preds/test_A_rf_d10_{raw,sub,concat}_seed{42,43,44}.csv` (test는 채점 없음) |

결과 파일은 2026-11-22 Kaggle 실행의 원본 출력이다. 9.2의 seed 확률 평균 BA와 9.4의 재현율·오분류는 저장된 val 예측 파일을 공통 `evaluate.py`로 다시 계산한 값이다.
