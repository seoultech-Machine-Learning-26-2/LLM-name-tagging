# B · ModernBERT-base + 대조 학습 실험 결과 (B-5, B-6)

> 담당: 정혜원(B) · 상태: **val 실험 완료(seed 42 1회) / test 미평가** · 작성: 2026-11-18 · 정리: B/Claude

코드·결과는 `hyewon` 브랜치에 있다. 아래 숫자는 모두 **val(질문 22개 × 18모델 = 396개)** 결과이며 test는 열지 않았다.

## 1. 실행 환경

| 항목 | 내용 |
| --- | --- |
| 실행 위치 | Google Colab, Tesla T4 (15GB) |
| Python / PyTorch / transformers | 3.13.16 / 2.14.1+cu130 / 5.23.0 |
| NumPy / pandas / scikit-learn | 2.4.6 / 2.2.3 / 1.6.1 |
| 체크포인트 | `answerdotai/ModernBERT-base` (Hugging Face, revision 고정 안 함) |
| 코드 커밋 | `hyewon` [`86e3aff`](https://github.com/seoultech-Machine-Learning-26-2/LLM-name-tagging/tree/86e3aff) (결과 커밋 `2ed97fc`) |
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

### 6.3 템플릿 표 (대조 λ=0.1)

| 평가 지표 | 개발사 6개 Validation | 18개 모델 Validation | Test |
| --- | --- | --- | --- |
| Accuracy | 0.7121 | 0.3106 | 미평가 |
| Precision (Macro) | 0.6649 | 0.2854 | 미평가 |
| Recall (Macro) = BA | 0.5277 | 0.3106 | 미평가 |
| F1 (Macro) | 0.5353 | 0.2801 | 미평가 |
| F1 (Weighted) | 0.6734 | 0.2801 | 미평가 |
| Log loss / ECE | 0.8518 / 0.0595 | 1.9962 / 0.0463 | 미평가 |
| 형제 정확도 | – | 0.4393 | 미평가 |

## 7. 오류 분석

- **개발사 혼동(대조 λ=0.1, 행 정규화):** Google 0.97, Anthropic 0.83, OpenAI 0.83은 높다. **ZAI(GLM)의 68%를 Anthropic으로**, Moonshot(Kimi)의 64%와 Qwen의 50%를 OpenAI로 예측한다. SVM(결합)과 같은 패턴이라, 모델 종류보다 데이터 자체의 유사성(아포스트로피·대시 습관 등, [B-svm](B-svm.md) 7절)에서 온 혼동일 가능성이 크다.
- SVM 결합(C=0.1)과 비교하면 18개 BA는 비슷하지만(0.301 vs 0.311) 개발사 BA(0.456 vs 0.528)와 Log loss(2.160 vs 1.996)는 ModernBERT가 낫다. 두 실행 모두 seed 1개라 차이를 확정하지 않는다.
- 사례별 오분류 표는 아직 만들지 않았다(예측 파일로 생성 가능).

## 8. 결론·한계

- ModernBERT 기본형은 4 epoch(약 3.5분)에 val BA 0.293, 대조 학습은 0.311로 수치상 높지만 **질문 단위 구간상 차이가 불확실**하다. 임베딩 분리도(Silhouette)는 나아지지 않았으므로 "대조 학습이 작성자 표현을 개선했다"고 쓰지 않는다.
- 한계: seed 1개(42), 4 epoch에서 아직 상승 중(학습 부족 가능), λ·τ·lr 미튜닝, 체크포인트 revision 미고정, 512토큰 초과 부분은 잘림, 모델 가중치는 저장하지 않음.
- 남은 선택 실험(11/18 이후 배정 없음): epoch 증가, seed 43/44 반복으로 변동 확인.
- **test 평가 후보(미평가, 팀 확인 후 1회):** 기본형 λ=0, 대조 λ=0.1.

## 9. 코드·산출물

| 산출물 | 경로 (`hyewon` 브랜치) |
| --- | --- |
| 학습 코드 | [`experiments/B_modernbert.py`](https://github.com/seoultech-Machine-Learning-26-2/LLM-name-tagging/blob/hyewon/experiments/B_modernbert.py) (`supcon`, `pk_batches` 자체 검사 포함) |
| 실행 노트북 | [`notebooks/B_colab.ipynb`](https://github.com/seoultech-Machine-Learning-26-2/LLM-name-tagging/blob/hyewon/notebooks/B_colab.ipynb), Kaggle용 [`B_kaggle.ipynb`](https://github.com/seoultech-Machine-Learning-26-2/LLM-name-tagging/blob/hyewon/notebooks/B_kaggle.ipynb) (미실행) |
| epoch 로그·요약 | [`results/course/B/modernbert/`](https://github.com/seoultech-Machine-Learning-26-2/LLM-name-tagging/tree/hyewon/results/course/B/modernbert) (`epoch_log.csv`, `summary.json`) |
| val 예측 (C `save_pred` 형식) | [`preds/val_B_modernbert_lam0_seed42.csv`, `..._lam0.1_seed42.csv`](https://github.com/seoultech-Machine-Learning-26-2/LLM-name-tagging/tree/hyewon/preds) |
| 체크포인트 | 저장 안 함 — Colab T4에서 실행당 약 4분이라 같은 명령으로 재학습 (GPU 비결정성으로 숫자가 조금 다를 수 있음) |
