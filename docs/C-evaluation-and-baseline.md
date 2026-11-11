# [C] 평가 지표 및 코드 구축

> 상태: 완료 · 남은 일: 팀 합의 사항(7번) · 담당: 서린 양(C) · 문서 이관: 2026-11-11 · B/Codex

## 1. 완료한 일과 근거

- 주 지표로 Balanced Accuracy(BA, 클래스별 재현율의 평균)를 정리했다.
- 예측 파일 검사·공통 평가·혼동행렬·질문 단위 bootstrap 코드를 구현했다.
- A 분할로 문자 TF-IDF+Logistic Regression 기준선을 만들고 val/test 예측과 지표를 저장했다.
- 11/11 Colab에서 A 분할로 기준선을 다시 학습해 지표·신뢰구간·개발사별 재현율이 모두 일치함을 확인했다. 이번 문서 이관에서는 저장된 GitHub 코드·결과 CSV와 숫자를 대조했으며, 학습을 다시 실행하지 않았다.
- 같은 질문 안에서 1:1로 배정하는 헝가리안 후처리를 기준선에 적용해 val 효과를 확인했다.

자료는 노션 기록과 [C 브랜치](https://github.com/seoultech-Machine-Learning-26-2/LLM-name-tagging/tree/eval/baseline-v2)다.

## 2. 지표와 공통 코드

18개 모델은 표본 수가 거의 같아 BA와 Accuracy가 유사하다. 개발사는 보유 모델 수가 달라 단순 Accuracy만 보면 소수 개발사 오류를 놓칠 수 있다.

| 지표 | 설명 |
| --- | --- |
| BA | 각 클래스 재현율을 같은 비중으로 평균 |
| Accuracy | 전체 응답 중 정답 비율 |
| Macro Precision/Recall/F1 | 클래스별 지표를 동일 가중 평균 |
| Weighted F1 | 응답 수로 가중한 F1 |
| Log loss | 정답에 부여한 확률의 품질; 낮을수록 좋음 |
| ECE | 예측 확신도와 실제 정답률 차이; 낮을수록 좋음 |
| 형제 정확도 | 18개 중 1위 모델의 개발사를 맞힌 응답 중 모델도 맞힌 비율; 모델이 1개인 개발사는 제외 |

| 함수 | 역할 |
| --- | --- |
| `load_data` | A 분할의 행 수·고유 ID·질문 겹침 검사 |
| `load_run` | `id + CLASSES` 열, 평가 대상 ID, 유한·비음수 확률, 행 합 검사 |
| `score`, `evaluate` | 18개 모델·6개 개발사 지표, seed 평균·표준편차 |
| `bootstrap_ci`, `compare` | 질문 단위 재표집 1,000회로 95% 신뢰구간·개선 비교 |
| `confusion` | 실제 클래스 기준 정규화 혼동행렬 |
| `leaderboard`, `template_table` | 비교표·보고서 표 생성 |
| `assign` | 같은 질문 안에서 응답 1개당 모델 1개로 배정 (헝가리안, 노션 기록) |

같은 질문의 응답 18개는 독립 표본으로 보기 어려워 bootstrap에서도 질문 묶음을 함께 뽑는다. `compare`의 차이 신뢰구간 하한이 0보다 클 때 코드상 `improved=True`다. 이는 기존 개선 판정 기능이며, 다음 강의 중심 과제에서 모든 모델이 반드시 통과해야 한다는 뜻은 아니다.

예측 CSV는 첫 열 `id`, 나머지 열은 `CLASSES`의 18개 확률이다. A의 알파벳순 라벨과 맞춘다. 행 합은 1이어야 하고 평가 대상 ID가 정확히 일치해야 한다. test 로드는 `open_test=True`를 명시해야 한다.

## 3. 기준선 설정

| 항목 | 저장된 노트북 설정 |
| --- | --- |
| 입력 | 원문 `text`, A 공통 분할 |
| TF-IDF | 문자 2~6그램, `sublinear_tf=True`, `min_df=3`, `max_features=400000` |
| 대소문자 | `TfidfVectorizer` 기본값 `lowercase=True` |
| 분류기 | `LogisticRegression(C=10, max_iter=3000)` |
| 전처리 학습 | train에만 `fit_transform`, val/test에는 `transform` |
| 학습 데이터 | train만 사용 |
| 실행 기록 | Colab, seed 1개. Python 3.13.18, numpy 2.4.6, pandas 2.2.3, scikit-learn 1.6.1, scipy 1.16.3 |

B의 새 TF-IDF 설정(문자 3~5그램·최대 30,000차원·대소문자 유지)과 다르다. C의 기존 기준선에 B 설정을 적용한 결과라고 쓰지 않는다. 팀 기준선은 회의에서 확정한다 (C 제안: 현재 기준선을 고정하고, B 설정은 대소문자 유지 효과를 보는 실험으로 기준선과 비교).

## 4. 저장된 평가 결과

### 18개 모델

| 지표 | Validation | Test |
| --- | ---: | ---: |
| BA = Macro Recall | 0.2904 | 0.3019 |
| Accuracy | 0.2904 | 0.3019 |
| Macro Precision | 0.2949 | 0.3064 |
| Macro F1 | 0.2810 | 0.2925 |
| Weighted F1 | 0.2810 | 0.2925 |
| Log loss | 2.3244 | 2.3255 |
| ECE | 0.1401 | 0.1487 |
| 형제 정확도 | 0.4015 | 0.4081 |

### 6개 개발사

| 지표 | Validation | Test |
| --- | ---: | ---: |
| BA — 확률 합산 | 0.4523 | 0.4362 |
| BA — 1위 모델의 개발사 | 0.5201 | 0.5565 |
| Accuracy — 확률 합산 | 0.7020 | 0.6787 |
| Macro Precision | 0.3630 | 0.3525 |
| Macro Recall — 확률 합산 | 0.4523 | 0.4362 |
| Macro F1 | 0.4005 | 0.3866 |
| Weighted F1 | 0.6174 | 0.5963 |
| Log loss | 1.1691 | 1.1694 |
| ECE | 0.2798 | 0.2599 |

`company_BA`는 같은 개발사의 모델 확률을 합산한 뒤 개발사를 고른다. `company_BA_argmax`는 18개 중 1위 모델의 개발사를 사용한다. 서로 다른 규칙이므로 한 이름의 점수처럼 섞지 않는다. Log loss·ECE는 합산 개발사 확률을 사용한다.

| BA 95% 신뢰구간 | Validation | Test |
| --- | --- | --- |
| 모델 18개 | 0.2374~0.3434 | 0.2609~0.3406 |
| 개발사 6개 — 합산 | 0.4295~0.4712 | 0.4112~0.4587 |

### 결과 해석

- 모델 BA는 균등 무작위 선택의 기대값 약 0.056보다 높지만, 세부 모델 구분에는 여전히 어려움이 있다.
- val 형제 정확도 0.4015는 개발사를 맞힌 다중 모델 개발사 응답 중 약 60%에서 세부 모델을 틀렸다는 의미다.
- val 합산 개발사 재현율은 Anthropic 0.8273, Google 0.9318, OpenAI 0.9545이며 ZAI·Moonshot·Qwen은 모두 0이다. 높은 전체 정확도만으로 모든 개발사를 잘 맞힌다고 판단할 수 없다.
- val과 test는 서로 다른 질문이므로 점수를 직접 비교해 성능이 개선됐다고 쓰지 않는다. 작은 검증 질문 수와 넓은 신뢰구간을 함께 고려한다.

### 헝가리안 배정 효과 (val)

같은 질문의 응답 18개에 18개 모델을 하나씩 겹치지 않게 배정하는 후처리다. 모델을 다시 학습하지 않고 기준선 확률 파일에 적용했다. test는 열지 않았다.

| 지표 (val) | 기준선 | + 헝가리안 | 변화 |
| --- | ---: | ---: | ---: |
| 18개 모델 BA | 0.2904 | 0.4040 | +0.1136 |
| 형제 정확도 | 0.4015 | 0.5242 | +0.1227 |
| 개발사 BA — 1위 모델의 개발사 | 0.5201 | 0.6208 | +0.1007 |

- paired bootstrap 차이 95% 신뢰구간: 모델 BA 0.0631~0.1641, 개발사 BA(합산) 0.1140~0.2273로 둘 다 하한이 0보다 크다.
- "같은 질문의 응답 18개가 함께 있다"는 가정을 쓰므로 배정 전과 배정 후 점수를 따로 기록한다.
- 배정 후 확률은 0 또는 1이 되므로 Log loss·ECE는 해석하지 않는다.

## 5. 원문에 정리된 실험 규칙

train 학습, val 튜닝·조기 종료, 설정 고정 후 test 평가가 원칙이다. 원문은 최종 후보를 seed 3개 평균 val BA로 선택하고, 1위와의 차이가 1위의 seed 표준편차 이내면 log loss가 낮은 쪽을 선택하도록 제시했다. test 예측은 train만으로 학습한 모델을 사용한다.

고전 모델은 공통 문체 입력 비교와 모델별 특징 확장 비교를 나누고, SVM 확률 보정도 train 내부 `prompt_id` 그룹 분할로 수행한다. 스케일러·TF-IDF는 train에만 fit한다. 새로운 계획에서 변경할 기준은 팀이 확인해 기록한다.

## 6. 코드·결과 위치

아래 파일은 문서 작성 시 C 브랜치에 있다. 문서 이관은 코드 병합을 포함하지 않는다.

- [평가 코드 `eval/evaluate.py`](https://github.com/seoultech-Machine-Learning-26-2/LLM-name-tagging/blob/eval/baseline-v2/eval/evaluate.py)
- [베이스라인 노트북](https://github.com/seoultech-Machine-Learning-26-2/LLM-name-tagging/blob/eval/baseline-v2/eval/baseline_tfidf.ipynb): 개인 Google Drive 경로는 실행 환경에 맞게 바꿔야 한다.
- [예측 CSV 2개](https://github.com/seoultech-Machine-Learning-26-2/LLM-name-tagging/tree/eval/baseline-v2/preds)
- [지표·신뢰구간·보고서 표 CSV 4개](https://github.com/seoultech-Machine-Learning-26-2/LLM-name-tagging/tree/eval/baseline-v2/results)

## 7. 남은 확인

- 팀 기준선 확정: C 기준선 유지 여부와 B TF-IDF 설정의 위치(기준선 또는 실험)
- 공통 문체 특징 채택, 개발사 예측 방식(확률 합산 / 1위 모델의 개발사), 최종 평가 시점·재학습 규칙
- 헝가리안 배정 기록 방식: 배정 전 BA를 주 지표로 두고 배정 후 BA는 별도 기록할지
- D 원본 템플릿의 모델 선택 기준(Macro F1)과 C 기준(BA) 일치
- 정확한 실행 버전 기록과 C 코드의 팀 기준 브랜치 통합은 별도 작업이다.
