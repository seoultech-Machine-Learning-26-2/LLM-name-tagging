# B · ModernBERT-base + 대조 학습 실험 결과 (B-5, B-6)

> 담당: 정혜원(B) · 상태: **val 실험 완료(4에폭 seed 42 + 8에폭 seed 42·43·44) / test 예측 저장(8에폭, 채점은 팀 통합 때 1회)** · 작성: 2026-11-18 · 수정: 2026-11-18 · 정리: B/Claude

코드·결과는 `hyewon` 브랜치에 있다. 아래 숫자는 모두 **val(질문 22개 × 18모델 = 396개)** 결과다.

> **test 채점 보류 (2026-11-18):** 이 문서는 처음에 당시 계획(11/17~11/18 각자 test 1회 평가)에 따라 test를 한 번 채점해 숫자를 적었다(`hyewon` `cd6d620`). 같은 날 팀 앙상블과 test 일괄 채점으로 계획을 바꾸면서([다음 작업 계획](../next-tasks-A-B-C-D.md) 4절) test 숫자는 이 문서에서 빼고, 공식 test 결과는 통합 채점값을 쓴다. 채점 뒤 설정·선택을 바꾼 것은 없으며, test 예측 파일은 그대로 제출한다.

## 1. 실행 환경

| 항목 | 내용 |
| --- | --- |
| 실행 위치 | Google Colab, Tesla T4 (15GB) |
| Python / PyTorch / transformers | 3.13.16 / 2.14.1+cu130 / 5.23.0 |
| NumPy / pandas / scikit-learn | 2.4.6 / 2.2.3 / 1.6.1 |
| 체크포인트 | `answerdotai/ModernBERT-base` revision `8949b909ec900327062f0ebf497f51aef5e6f0c8` (실행 당시 코드는 revision을 지정하지 않았으나, 이 revision이 2025-01-15 이후 Hub의 최신이라 같은 것을 받았다. `2658da7`부터 코드에 고정) |
| 코드 커밋 | 4에폭: `hyewon` [`86e3aff`](https://github.com/seoultech-Machine-Learning-26-2/LLM-name-tagging/tree/86e3aff) (결과 커밋 `2ed97fc`) · 8에폭·test: [`109f514`](https://github.com/seoultech-Machine-Learning-26-2/LLM-name-tagging/tree/109f514) (결과 커밋 `1c5dc2a`) |
| 실행 명령 | `python -m experiments.B_modernbert --lam 0` / `--lam 0.1` ([Colab 노트북](https://github.com/seoultech-Machine-Learning-26-2/LLM-name-tagging/blob/hyewon/notebooks/B_colab.ipynb) 4번) |

## 2. 문제와 가설

- 같은 문제(18개 작성 LLM 분류, 질문 단위 분할)를 사전학습 언어모델로 푼다.
- 가설: 분류 손실에 **지도 대조 손실(SupCon)** 을 더해 "같은 작성 모델의 다른 질문 응답"을 임베딩 공간에서 가깝게 당기면, 질문 내용보다 작성자 문체에 기대는 표현을 배워 처음 보는 질문에서 BA가 오른다.

## 3. 이론 요약

- **ModernBERT-base**(약 1.5억 파라미터)는 양방향 Transformer 인코더다. 토큰 표현을 평균 풀링해 문서 임베딩을 만들고 선형층으로 18개 확률(softmax)을 낸다.
- **분류 손실:** 교차 엔트로피 — 정답 클래스 확률의 −log를 최소화.
- **SupCon 손실(Khosla et al., 2020):** 배치 안에서 정규화한 임베딩끼리 코사인 유사도/τ로 softmax를 만들고, 같은 라벨(양성)의 확률을 높인다. 즉 같은 작성자는 가깝게, 다른 작성자는 멀게 한다.
- **전체 목적함수:** `L = CE + λ · SupCon` (λ=0이면 기본형). 강의의 손실 최소화·일반화 관점에서, 대조 항은 "질문이 달라도 작성자가 같으면 같은 표현"이라는 제약을 추가하는 규제로 볼 수 있다.

## 4. 데이터·전처리

- 입력: 응답 원문 → ModernBERT 토크나이저, 최대 512토큰에서 자름, 배치별 동적 padding. 별도 정제 없음.
- **배치 구성(P×K):** 한 배치 = 작성 모델 8개 × 2개 = 16개. 모든 표본이 배치 안에 양성 1개 이상을 갖는다. 이 분할에서는 질문마다 모델별 답이 1개라 같은 작성자 쌍은 **항상 서로 다른 질문**에서 온다. 두 조건 모두 같은 샘플러를 써서 배치 구성 차이를 없앴다.

## 5. 설정·통제 조건

| 항목 | 기본형 (B-5) | 대조 학습 (B-6) |
| --- | --- | --- |
| λ (SupCon 가중치) | 0 | 0.1 |
| τ (온도) | (기록만) 0.1 | 0.1 |
| 공통 | AdamW lr 3e-5, wd 0.01, warmup 10% 선형 감소, 4 epoch, fp16 autocast, grad clip 1.0, seed 42, 전체 파라미터 학습 | 동일 |
| epoch 선택 | val 모델 BA 최고 epoch | 동일 |

시작값이며 튜닝하지 않았다. 기본형에서도 SupCon 값은 계산·기록만 하고 역전파하지 않는다.

## 6. train / val 결과

### 6.1 epoch별 (val)

| epoch | 기본형 train CE | 기본형 val BA | 대조 train CE | 대조 train SupCon | 대조 val BA |
| --- | --- | --- | --- | --- | --- |
| 1 | 2.839 | 0.109 | 2.812 | 2.634 | 0.141 |
| 2 | 2.249 | 0.227 | 2.253 | 2.266 | 0.210 |
| 3 | 1.858 | 0.258 | 1.819 | 2.045 | 0.245 |
| 4 | 1.549 | **0.293** | 1.457 | 1.728 | **0.311** |

두 실행 모두 **마지막 epoch이 최고**였고 val BA가 계속 오르는 중이었다. 따라서 epoch 선택이 val 점수를 부풀린 효과는 없지만, **학습이 덜 된 상태**일 수 있다.

### 6.2 기본형 vs 대조 학습 (val, 최종 epoch)

| 지표 | 기본형 λ=0 | 대조 λ=0.1 | 변화 |
| --- | --- | --- | --- |
| 18개 모델 BA (=Acc) | 0.2929 | 0.3106 | +0.018 |
| 18개 Precision (Macro) | 0.2845 | 0.2854 | +0.001 |
| 18개 Macro F1 | 0.2661 | 0.2801 | +0.014 |
| 18개 Log loss / ECE | 1.956 / 0.058 | 1.996 / 0.046 | |
| 형제 정확도 | 0.400 | 0.439 | +0.039 |
| 개발사 BA (확률합) | 0.5011 | 0.5277 | +0.027 |
| 개발사 Accuracy | 0.6894 | 0.7121 | +0.023 |
| 개발사 Macro F1 | 0.4867 | 0.5353 | +0.049 |
| 개발사 Log loss / ECE | 0.824 / 0.064 | 0.852 / 0.060 | |
| val 임베딩 Silhouette (cos) | −0.143 | −0.155 | −0.012 |
| 학습 시간 / 최대 GPU 메모리 | 211초 / 6.76GB | 220초 / 6.76GB | |
| 학습 파라미터 | 149,028,114 | 149,028,114 | |

- **불확실성 (질문 단위 paired bootstrap, 1000회):** 대조 − 기본형 모델 BA +0.017 [−0.023, 0.053], 개발사 BA +0.027 [−0.013, 0.064]. **구간이 0을 포함하므로 개선이라고 결론 내리지 않는다.**
- 각 실행의 모델 BA 95% CI: 기본형 0.250~0.338, 대조 0.258~0.366.
- **Silhouette은 오히려 약간 낮아졌다.** 대조 손실이 val 임베딩을 작성자별로 더 잘 모았다는 증거는 없다. 음수라서 val 임베딩은 작성자보다 다른 요인(질문 주제 등)으로 모여 있을 가능성이 크다.

### 6.3 템플릿 표 (4에폭 대조 λ=0.1, val만)

| 평가 지표 | 개발사 6개 Validation | 18개 모델 Validation |
| --- | --- | --- |
| Accuracy | 0.7121 | 0.3106 |
| Precision (Macro) | 0.6649 | 0.2854 |
| Recall (Macro) = BA | 0.5277 | 0.3106 |
| F1 (Macro) | 0.5353 | 0.2801 |
| F1 (Weighted) | 0.6734 | 0.2801 |
| Log loss / ECE | 0.8518 / 0.0595 | 1.9962 / 0.0463 |
| 형제 정확도 | – | 0.4393 |

4에폭에서 학습이 덜 끝났으므로 test 예측은 이 설정이 아니라 6.4절의 8에폭 설정으로 저장했다.

### 6.4 8에폭 × seed 3개 (val)

4에폭에서 BA가 계속 오르고 seed가 1개뿐이어서, **epoch만 8로 늘리고** 나머지 설정은 그대로 seed 42·43·44를 반복했다(`--epochs 8 --save_test`, Colab T4, 실행당 약 8분). 4에폭 결과와 별도 실험으로 기록한다.

| 조건 | seed 42 | seed 43 | seed 44 | 평균 ± 표준편차 | 개발사 BA 평균 ± 표준편차 | 선택 epoch | Silhouette(cos) 평균 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 기본형 λ=0 | 0.364 | 0.326 | 0.333 | **0.341 ± 0.020** | 0.582 ± 0.004 | 8, 8, 8 | −0.083 |
| 대조 λ=0.1 | 0.369 | 0.333 | 0.318 | **0.340 ± 0.026** | 0.575 ± 0.042 | 8, 5, 8 | −0.105 |

- 8에폭으로 늘리자 두 조건 모두 val BA가 4에폭(0.29~0.31)보다 올랐다. 다만 대부분 8에폭이 최고라 **여전히 수렴 전일 수 있다.**
- seed 간 표준편차(0.02~0.03)가 두 조건의 평균 차이(0.001)보다 훨씬 크다. **4에폭에서 보였던 대조 학습의 +0.018은 seed 변동 범위 안이었다.**
- Silhouette은 여전히 음수이고 대조 학습 쪽이 더 낮다.

### 6.5 test

각 seed에서 val로 고른 epoch의 test 확률만 저장했다(`--save_test`, 6개 파일). 채점은 팀 통합 때 C가 1회 한다. B의 최종 후보를 하나 고른다면 팀 선택 규칙(seed 평균 val BA, 1위와 차이가 표준편차 이내면 LogLoss가 낮은 쪽)에 따라 **기본형 λ=0**이다(val BA 0.341 vs 0.340, LogLoss 2.143 vs 2.170).

## 7. 오류 분석

- **개발사 혼동(대조 λ=0.1, 행 정규화):** Google 0.97, Anthropic 0.83, OpenAI 0.83은 높다. **ZAI(GLM)의 68%를 Anthropic으로**, Moonshot(Kimi)의 64%와 Qwen의 50%를 OpenAI로 예측한다. SVM(결합)과 같은 패턴이라, 모델 종류보다 데이터 자체의 유사성(아포스트로피·대시 습관 등, [B-svm](B-svm.md) 7절)에서 온 혼동일 가능성이 크다.
- SVM 결합(C=0.1)과 비교하면 18개 BA는 비슷하지만(0.301 vs 0.311) 개발사 BA(0.456 vs 0.528)와 Log loss(2.160 vs 1.996)는 ModernBERT가 낫다. 두 실행 모두 seed 1개라 차이를 확정하지 않는다.
- 사례별 오분류 표는 아직 만들지 않았다(예측 파일로 생성 가능).

## 8. 결론·한계

- ModernBERT 기본형은 4 epoch(약 3.5분)에 val BA 0.293, 대조 학습은 0.311로 수치상 높지만 **질문 단위 구간상 차이가 불확실**하다. 임베딩 분리도(Silhouette)는 나아지지 않았으므로 "대조 학습이 작성자 표현을 개선했다"고 쓰지 않는다.
- 8에폭 seed 3개로 다시 보면 두 조건의 val BA는 0.341 vs 0.340으로 거의 같다. **이 설정(λ=0.1, τ=0.1)의 SupCon은 일반화 성능을 올리지 않았다.**
- SVM 결합(val 0.301 / 개발사 0.456)과 비교하면 ModernBERT가 개발사 수준에서 크게 높다(0.58). test 비교는 통합 채점 뒤에 한다.
- 한계: seed 3개, 8에폭에서도 대부분 마지막 epoch이 최고(학습 부족 가능), λ·τ·lr 미튜닝, 확률 보정 안 함(ECE 높음), 512토큰 초과 부분은 잘림, 모델 가중치는 저장하지 않음.

## 9. 코드·산출물

| 산출물 | 경로 (`hyewon` 브랜치) |
| --- | --- |
| 학습 코드 | [`experiments/B_modernbert.py`](https://github.com/seoultech-Machine-Learning-26-2/LLM-name-tagging/blob/hyewon/experiments/B_modernbert.py) (`supcon`, `pk_batches` 자체 검사 포함) |
| 실행 노트북 | [`notebooks/B_colab.ipynb`](https://github.com/seoultech-Machine-Learning-26-2/LLM-name-tagging/blob/hyewon/notebooks/B_colab.ipynb), Kaggle용 [`B_kaggle.ipynb`](https://github.com/seoultech-Machine-Learning-26-2/LLM-name-tagging/blob/hyewon/notebooks/B_kaggle.ipynb) (미실행) |
| epoch 로그·요약 | [`results/course/B/modernbert/`](https://github.com/seoultech-Machine-Learning-26-2/LLM-name-tagging/tree/hyewon/results/course/B/modernbert) (`epoch_log.csv`, `summary.json`) |
| val·test 예측 (C `save_pred` 형식) | [`preds/val_B_modernbert_lam{0,0.1}_seed42.csv`, `{val,test}_B_modernbert_lam{0,0.1}_ep8_seed{42,43,44}.csv`](https://github.com/seoultech-Machine-Learning-26-2/LLM-name-tagging/tree/hyewon/preds) |
| 모델 가중치 | 보고한 실행들은 저장하지 않았다. 같은 명령으로 재학습할 수 있다(실행당 4에폭 약 4분, 8에폭 약 8분, GPU 비결정성으로 숫자가 조금 다를 수 있음). `2658da7`부터 `--save_model <Drive 폴더>`를 주면 val로 고른 epoch 가중치를 `<폴더>/<run>_seed<seed>.pt`(fp16, 약 300MB)로 저장한다. 복원: `net = Net(); net.load_state_dict({k: v.float() for k, v in torch.load(경로).items()})` |
