# B · Linear SVM 실험 결과 (B-1 ~ B-4)

> 담당: 정혜원(B) · 상태: **val 실험 완료 / test 1회 평가 완료** · 작성: 2026-11-18 · 수정: 2026-11-18 · 정리: B/Claude

코드·결과는 `hyewon` 브랜치에 있다. 6.4절 외의 숫자는 모두 **val(질문 22개 × 18모델 = 396개)** 결과다. test(질문 23개 × 18모델 = 414개)는 설정을 고정한 뒤 6.4절에서 한 번만 채점했다.

## 1. 실행 환경

| 항목 | 내용 |
| --- | --- |
| 실행 위치 | Google Colab (Linux-6.6.159 x86_64), CPU 사용 |
| Python / NumPy / pandas / scikit-learn | 3.13.16 / 2.4.6 / 2.2.3 / 1.6.1 |
| 코드 커밋 | `hyewon` [`86e3aff`](https://github.com/seoultech-Machine-Learning-26-2/LLM-name-tagging/tree/86e3aff) (결과 커밋 `140ae05`) |
| 실행 명령 | `python -u -W ignore -m experiments.B_svm` ([Colab 노트북](https://github.com/seoultech-Machine-Learning-26-2/LLM-name-tagging/blob/hyewon/notebooks/B_colab.ipynb) 2번 셀) |
| seed | 42 (LinearSVC는 결정적이라 seed 반복 안 함) |

## 2. 문제와 가설

- 처음 보는 질문의 응답에서 작성 LLM 18개를 고른다. 같은 질문의 답이 train/val에 섞이지 않는다(A 분할).
- 가설 1: 문체 32개만으로도 무작위(1/18 ≈ 0.056)보다 훨씬 낫다.
- 가설 2: 문자 TF-IDF는 표현이 풍부하지만 질문 주제를 외워 새 질문에서 덜 일반화된다.
- 가설 3: 두 입력을 합치면 각각보다 낫다.

## 3. 이론 요약

선형 SVM은 클래스 사이 **마진(경계와 가장 가까운 점 사이 거리)을 최대화**하는 초평면을 찾는다. 목적함수는 `½‖w‖² + C·Σ hinge loss`이고 18클래스는 one-vs-rest로 18개 경계를 학습한다. C가 크면 학습 오분류에 벌점이 커져 경계가 복잡해지고(규제 약함), 작으면 단순해진다. 거리 기반이라 **특징 크기(스케일)에 민감**하므로 train으로 fit한 StandardScaler를 쓴다. SVM은 확률을 내지 않으므로 C 평가 코드용 확률은 sigmoid 보정으로 만든다.

## 4. 데이터·전처리

| 입력 | 차원 | 전처리 (모두 train에만 fit) |
| --- | --- | --- |
| 문체 (`style`) | 32 | B 공통 특징 → StandardScaler(**평균 중심화**) |
| 문자 TF-IDF (`tfidf`) | 30,000 | B 공통 `make_extractor("tfidf")`: char 3~5gram, min_df=2, sublinear |
| 결합 (`combined`) | 30,032 | 문체(중심화 스케일) + 문자 TF-IDF |
| 문체·스케일링 없음 (`style_raw`) | 32 | 원시 값 (스케일링 효과 확인용) |

- **B-1 입력 확인:** train에서 32개 중 9개가 상수(`line_count`, `blank_line_ratio`, heading/bullet/numbered/code_fence/bold/link/backtick 비율). 입력 파일에 줄바꿈·Markdown이 거의 없기 때문이며, 원래 응답에 형식이 없었다고 단정하지 않는다. 공통 비교를 위해 32개를 그대로 둔다.
- **공통 코드와 다른 점:** 공통 `make_extractor`는 sparse 출력 때문에 `with_mean=False`(평균 미중심화)다. 결합 입력에서 이 방식은 liblinear 학습 1회에 약 173초가 걸려, B 실험에서는 문체 블록만 평균 중심화했다. 공통 코드(`features/`)는 바꾸지 않았다.

## 5. 설정·통제 조건

- `LinearSVC(C, max_iter=5000, random_state=42)`, C ∈ {0.1, 1, 10} — 입력 3종 × 3 = 9개 + 스케일링 비교 1개 + Dummy 기준선.
- **C\* 선택:** 문체 입력의 val BA(결정함수 argmax)가 가장 높은 C. 세 C 모두 0.2778로 같아 가장 단순한(규제가 강한) **C\*=0.1**을 택했다.
- **확률 보정:** `CalibratedClassifierCV(sigmoid)`, train 안에서 `prompt_id` 기준 GroupKFold(5). 각 fold 안에서 특징 추출부터 다시 fit한다. 보정 전 SVM 지표(`svm_*`)와 보정 후 확률 지표를 구분해 적는다.

## 6. train / val 결과

### 6.1 전체 설정 (val)

| 입력 | C | train BA | val BA (SVM) | val BA (보정) | Macro F1 (보정) | 개발사 BA (확률합) | Log loss | 수렴 | 학습 1회(초) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Dummy(prior) | – | – | – | 0.056 | 0.006 | 0.167 | 2.890 | – | – |
| 문체 | 0.1 | 0.319 | 0.278 | 0.258 | 0.231 | 0.410 | 2.288 | O | 1.0 |
| 문체 | 1 | 0.324 | 0.278 | 0.258 | 0.235 | 0.402 | 2.289 | O | 1.0 |
| 문체 | 10 | 0.326 | 0.278 | 0.255 | 0.232 | 0.406 | 2.294 | O | 2.0 |
| TF-IDF | 0.1 | **1.000** | 0.268 | 0.290 | 0.273 | 0.439 | 2.157 | O | 4.4 |
| TF-IDF | 1 | **1.000** | 0.270 | 0.288 | 0.264 | 0.442 | 2.241 | O | 6.5 |
| TF-IDF | 10 | **1.000** | 0.263 | 0.273 | 0.246 | 0.425 | 2.290 | O | 6.7 |
| 결합 | 0.1 | 0.857 | 0.301 | 0.301 | 0.271 | 0.456 | 2.160 | O | 67.6 |
| 결합 | 1 | 1.000 | 0.303 | 0.311 | 0.287 | 0.500 | 2.110 | **X** | 177.6 |
| 결합 | 10 | 1.000 | 0.298 | 0.313 | 0.289 | 0.487 | 2.130 | **X** | 200.9 |
| 문체·스케일링 없음 | 0.1 | 0.269 | 0.230 | 0.235 | 0.201 | 0.366 | 2.403 | O | 1.1 |

val은 클래스마다 22개로 균형이라 BA = Accuracy다. 원본: [`metrics.csv`](https://github.com/seoultech-Machine-Learning-26-2/LLM-name-tagging/blob/hyewon/results/course/B/svm/metrics.csv)

### 6.2 B-3 특징 비교 (C\*=0.1 고정)

| 입력 | val BA (SVM) | 95% CI (질문 단위 bootstrap, 보정 확률) |
| --- | --- | --- |
| 문체 | 0.278 | 0.210 ~ 0.303 |
| TF-IDF | 0.268 | – |
| 결합 | **0.301** | 0.250 ~ 0.351 |

- 결합 − 문체 (paired bootstrap, 보정 확률): 모델 BA +0.041 [0.008, 0.076], 개발사 BA +0.047 [0.023, 0.076] → 두 구간 모두 0보다 커서 이 val에서는 결합이 문체보다 낫다.
- 결합 − C의 TF-IDF+LR 기준선: 모델 BA +0.009 [−0.033, 0.056] → **기준선과 차이가 있다고 볼 수 없다.**
- 입력별로 C를 다시 고르면 결합 C=1(0.303)이 가장 높지만 C=0.1과 차이가 0.0025(1개 미만)이고 수렴하지 않았다. 별도 표: [`feature_compare_retunedC.csv`](https://github.com/seoultech-Machine-Learning-26-2/LLM-name-tagging/blob/hyewon/results/course/B/svm/feature_compare_retunedC.csv)

### 6.3 템플릿 표 (최종 후보: 결합 C=0.1, 보정 확률)

| 평가 지표 | 개발사 6개 Validation | 개발사 6개 Test | 18개 모델 Validation | 18개 모델 Test |
| --- | --- | --- | --- | --- |
| Accuracy | 0.6692 | 0.6594 | 0.3005 | 0.3140 |
| Precision (Macro) | 0.5499 | 0.5638 | 0.2621 | 0.2994 |
| Recall (Macro) = BA | 0.4557 | 0.4380 | 0.3005 | 0.3140 |
| F1 (Macro) | 0.4277 | 0.4117 | 0.2710 | 0.2936 |
| F1 (Weighted) | 0.6032 | 0.5936 | 0.2710 | 0.2936 |
| Log loss / ECE | 1.0240 / 0.1475 | 1.0719 / 0.1517 | 2.1600 / 0.0618 | 2.2789 / 0.0744 |
| 형제 정확도 | – | – | 0.4534 | 0.4574 |

개발사 지표는 18개 확률을 개발사별로 합산한 방식(`company_BA`)이다. 1위 모델의 개발사를 쓰는 `company_BA_argmax`(0.494)와 직접 비교하지 않는다.

### 6.4 test 1회 평가 (보정 확률)

val에서 정한 두 설정을 train 전체로 다시 학습해 test 확률을 저장했고(`python -m experiments.B_svm --test`, Colab, 코드 `109f514`), 채점은 [`experiments/B_test_eval.py`](https://github.com/seoultech-Machine-Learning-26-2/LLM-name-tagging/blob/hyewon/experiments/B_test_eval.py)로 한 번만 했다(결과 커밋 `cd6d620`). 결과를 보고 다시 튜닝하지 않는다.

| 입력 (C=0.1) | val 모델 BA | test 모델 BA [95% CI] | val 개발사 BA | test 개발사 BA [95% CI] |
| --- | --- | --- | --- | --- |
| 문체 | 0.258 | 0.278 [0.242, 0.316] | 0.410 | 0.413 [0.385, 0.442] |
| 결합 | 0.301 | **0.314** [0.268, 0.362] | 0.456 | 0.438 [0.400, 0.475] |

- 여기 BA는 보정 확률의 argmax라 6.1절 "val BA (SVM)" 열(문체 0.278)과 다르다.
- test에서도 결합 − 문체는 모델 BA +0.036 [0.007, 0.068], 개발사 BA +0.025 [0.003, 0.046]로 **val의 결론(결합이 낫다)이 유지**됐다. val과 test 차이는 구간 폭(±0.05) 안이다.
- 원본: [`results/course/B/test/`](https://github.com/seoultech-Machine-Learning-26-2/LLM-name-tagging/tree/hyewon/results/course/B/test) (`summary.csv`, `compare_test.csv`, `template_*.csv`)

## 7. 해석과 오류 분석 (B-4)

- **스케일링이 필요한 이유:** 같은 문체 32개에서 스케일링을 빼면 val BA 0.278 → 0.230. `char_count`(수백~수천)처럼 큰 값이 마진 계산을 지배해 작은 비율 특징이 무시되기 때문으로 해석한다.
- **TF-IDF는 train을 외운다:** 모든 C에서 train BA 1.000, val 0.26~0.27. 3만 차원에 1,889개 표본이라 train은 거의 항상 선형 분리 가능하다. 다만 상위 n-gram([`top_coef_tfidf_C0.1.csv`](https://github.com/seoultech-Machine-Learning-26-2/LLM-name-tagging/blob/hyewon/results/course/B/svm/top_coef_tfidf_C0.1.csv))은 주제어보다 **타이포그래피**가 많다: GPT 계열은 둥근 아포스트로피 `’s`, Claude·GLM은 곧은 `'s`, Claude Opus 4.6/5는 앞뒤 공백 있는 `— `, GLM은 공백 없는 `—th`. 일부(kimi `breath`, gemini-3.6 `human`)는 내용 선호일 수 있다. 주제 의존이 입증된 것은 아니다.
- **문체 상위 계수**([`top_coef_style_C0.1.csv`](https://github.com/seoultech-Machine-Learning-26-2/LLM-name-tagging/blob/hyewon/results/course/B/svm/top_coef_style_C0.1.csv)): Claude 계열은 `em_dash`, `whitespace_ratio`, `mean_word_length`(GLM도 `em_dash`), GPT-5.6·GPT-6 계열은 `non_ascii_ratio`(둥근 따옴표·특수문자)가 양의 계수. 상관된 특징끼리 계수가 나뉠 수 있어 크기만으로 원인을 단정하지 않는다.
- **개발사 혼동(결합 C=0.1):** Google 0.94, OpenAI 0.86, Anthropic 0.75는 잘 맞히지만 **ZAI(GLM)의 64%를 Anthropic으로**, Moonshot(Kimi)의 77%를 OpenAI로, Qwen의 59%를 OpenAI로 예측한다. 위의 아포스트로피·대시 습관이 GLM과 Claude에서 같다는 관찰과 일치한다.
- **18개 모델 재현율:** `claude-fable-5`는 세 입력 모두 0.00, `glm-5.3` 0.05로 거의 못 맞힌다. `gemini-3.1-pro-preview`(0.73), `gpt-6-astra`, `claude-opus-5`(0.59)는 높다. 결합 오류 277개 중 49%가 같은 개발사 안의 혼동이다. 그림: [`confusion_combined_C0.1.png`](https://github.com/seoultech-Machine-Learning-26-2/LLM-name-tagging/blob/hyewon/results/course/B/svm/confusion_combined_C0.1.png)

### 대표 오분류 (결합 C=0.1, [`errors_combined_C0.1.csv`](https://github.com/seoultech-Machine-Learning-26-2/LLM-name-tagging/blob/hyewon/results/course/B/svm/errors_combined_C0.1.csv))

| id | 질문 | 실제 → 예측 | 관찰 | 가설 |
| --- | --- | --- | --- | --- |
| 017623d68134 | p142 | gemini-3.8-flash → gemini-3.7-flash | 같은 개발사, 수식어 많은 서사체 | 같은 계열 flash 모델의 문체가 거의 같음 |
| 01b5eec535ac | p022 | claude-opus-4-8 → claude-sonnet-5 | 비유 많은 문장, `'s` 사용 | Claude 내부 구분 특징이 부족 |
| 016063f5ba29 | p019 | glm-5.3-flash → gpt-5.6-luna | 공백 없는 `—` 사용 | 과학 설명체에서 GLM/GPT 구분 신호가 약함 |
| 0041ac86ee93 | p073 | claude-sonnet-5 → gemini-3.7-flash | 짧은 나열식 설명 | 같은 질문(p073)에 답들이 비슷한 구조 |
| 009319596f10 | p073 | gpt-5.6-terra → claude-sonnet-5 | 쉼표로 긴 나열 | 질문이 정한 형식이 작성자 신호를 덮음 |

관찰은 응답 앞 300자를 보고 적은 것이며 가설은 검증하지 않았다.

## 8. 결론·한계

- 문체 32개만으로 val BA 0.278(무작위 0.056)이고, 결합하면 0.301로 오른다(문체 대비 개선 구간 > 0). 그러나 **C의 TF-IDF+LR 기준선(0.290)과는 차이가 없다.**
- 개발사 수준은 잘 구분하나 GLM↔Claude, Kimi·Qwen→OpenAI 혼동이 크다.
- 한계: val 질문 22개뿐이라 구간이 넓다(±0.05). 결합 C=1/10은 5,000회 안에 수렴하지 않았다. 문체 스케일링 방식이 공통 extractor와 다르다. 보정은 one-vs-rest sigmoid라 개발사 ECE(0.148)가 높다.
- **test(1회):** 결합 C=0.1(B 최종) 모델 BA 0.314, 개발사 BA 0.438. 문체 C=0.1(공통 입력 비교용) 0.278 / 0.413. ModernBERT와의 test 비교는 [B-modernbert](B-modernbert.md) 6.5절.

## 9. 코드·산출물

| 산출물 | 경로 (`hyewon` 브랜치) |
| --- | --- |
| 학습 코드 | [`experiments/B_svm.py`](https://github.com/seoultech-Machine-Learning-26-2/LLM-name-tagging/blob/hyewon/experiments/B_svm.py) |
| 실행 노트북 | [`notebooks/B_colab.ipynb`](https://github.com/seoultech-Machine-Learning-26-2/LLM-name-tagging/blob/hyewon/notebooks/B_colab.ipynb) |
| 지표·비교표·설정 | [`results/course/B/svm/`](https://github.com/seoultech-Machine-Learning-26-2/LLM-name-tagging/tree/hyewon/results/course/B/svm) (`metrics.csv`, `feature_compare_*.csv`, `run_meta.json`) |
| 혼동행렬·오류·계수 | 같은 폴더 `confusion_*`, `errors_*`, `top_coef_*` |
| test 채점 코드·결과 | [`experiments/B_test_eval.py`](https://github.com/seoultech-Machine-Learning-26-2/LLM-name-tagging/blob/hyewon/experiments/B_test_eval.py), [`results/course/B/test/`](https://github.com/seoultech-Machine-Learning-26-2/LLM-name-tagging/tree/hyewon/results/course/B/test) |
| val·test 예측 (C `save_pred` 형식) | [`preds/val_B_svm_*_seed42.csv`, `preds/test_B_svm_{style,combined}_C0.1_seed42.csv`](https://github.com/seoultech-Machine-Learning-26-2/LLM-name-tagging/tree/hyewon/preds) |
| 체크포인트 | 저장 안 함 (CPU 수 분 내 재학습 가능) |
