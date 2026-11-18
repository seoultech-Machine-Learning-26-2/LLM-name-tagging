# C · Qwen3.5-0.8B + LoRA 실험 결과 (C-4 ~ C-5)

> 담당: 양서린(C) · 상태: **val 실험 완료 / test 예측 저장(채점은 팀 통합 때 1회)** · 작성: 2026-11-18 · 정리: C/Claude

코드·예측은 `seorin` 브랜치에 있다. 아래 숫자는 모두 **val(질문 22개 × 18모델 = 396개)** 결과다. test 칸은 `미평가`로 둔다. 고전 모델은 [C · XGBoost](C-xgboost.md)에 있다.

> **질문 묶음 이용 표시:** Q3 이후 모든 설정은 **질문 중심화**(같은 `prompt_id` 응답 18개 임베딩의 평균을 빼서 함께 입력)를 쓴다. 예측할 때도 같은 질문의 응답 18개를 한 묶음으로 넣는다. 응답 단독 예측 결과(Q1·Q2)와 구분해 적는다([다음 작업 계획](../next-tasks-A-B-C-D.md) 4절 9번). 헝가리안 배정(후처리)은 주 지표에 넣지 않고 `배정 BA` 칸에만 적는다.

## 1. 실험 개요

| 항목 | 내용 |
| --- | --- |
| 담당자 | 양서린(C) |
| OS/CPU/RAM/GPU | Google Colab, NVIDIA T4 16GB (seed 42) → Kaggle Notebook, T4 ×2 중 1개 사용 (seed 43·44, 예측) |
| Python / NumPy / pandas / scikit-learn 버전 | [확인 필요] |
| 추가 사용 라이브러리 버전 | 실행할 때마다 `pip install -U transformers peft accelerate flash-linear-attention` (torchao 제거). 정확한 버전 [확인 필요] |
| 실험 모델 | `Qwen/Qwen3.5-0.8B` 텍스트 본체(`AutoModelForCausalLM(...).model`) + LoRA + 분류 헤드 |
| 모델 구분 | 심화 모델 |
| 실험 날짜 | 2026-11-15 ~ 2026-11-17 (seed 42, Colab), 2026-11-17 (seed 43·44, Kaggle), 2026-11-18 (앙상블·test 예측) |
| 실험 환경 | PyTorch fp32 가중치 + fp16 autocast, gradient checkpointing, flash-linear-attention |
| 코드 | `seorin` [`0434772`](https://github.com/seoultech-Machine-Learning-26-2/LLM-name-tagging/blob/0434772/experiments/C_qwen_lora.py) `experiments/C_qwen_lora.py` (학습) · [`experiments/C_ensemble_predict.py`](https://github.com/seoultech-Machine-Learning-26-2/LLM-name-tagging/blob/0434772/experiments/C_ensemble_predict.py) (앙상블·test 예측) |

## 2. 실험 설정

### 2.1 데이터 설정

| 항목 | 내용 |
| --- | --- |
| 데이터셋 | Name that LLM (2026) |
| 전체 데이터 | 2,699개 |
| Train / Validation / Test | 공통 분할 사용 (1,889 / 396 / 414) |
| 분할 기준 | `prompt_id` 기준 |
| Random Seed | 42 (분할) · 학습 seed 42/43/44 |
| 분류 대상 | 개별 모델 18개 직접 예측 → 개발사 6개는 확률 합산 |

### 2.2 특징 추출 및 전처리

| 항목 | 내용 |
| --- | --- |
| 입력 데이터 | `text` 원문 (질문 중심화에 `prompt_id`) |
| 전처리 방식 | Qwen 토크나이저, 최대 384토큰(데이터 최대 378토큰, 잘림 0%), 64의 배수로 패딩 |
| 특징 추출 방식 | 마지막 층 hidden state의 mean pooling → H (1024차원) |
| 주요 특징 | 질문 중심화: `[H ; H − 같은 질문 18개의 H 평균]` (2048차원) |
| 특징 차원 수 | 1024 (Q1·Q2) / 2048 (Q3 이후) |

### 2.3 모델 및 하이퍼파라미터

| **항목** | **기록** |
| --- | --- |
| 설정과 탐색 범위 | Q1 동결 → Q2 LoRA → Q3 +중심화 → 과적합 대응 V1(규제)·V2(문장 자르기 증강)·V3(R-Drop) → Q4 +계층형 헤드. 총 8설정 + seed 반복 |
| 고정한 조건 / 바꾼 조건 | 고정: LoRA r 16, α 32, dropout 0.05, 대상 `q_proj, k_proj, v_proj, o_proj, in_proj_qkv, in_proj_z, out_proj`(학습 파라미터 4,915,200개, 0.649%), AdamW(LoRA 2e-4, 헤드 1e-3, weight decay 0.01), warmup 10% + cosine, 3 epoch, gradient clip 1.0, 배치 = 질문 1개의 응답 18개 · 설정마다 한 가지만 추가 |
| 학습 seed | 탐색: 42 · 최종 후보 V3·Q4: 42/43/44 |
| 최종 설정 / 선택 이유 | **V3×3 + Q4×3 확률 평균 × 0.8 + XGBoost × 0.2.** val BA +1%p 이상일 때만 채택하는 규칙으로 V3 채택, Q4는 seed 3개 평균에서 V3보다 +1.9%p. 조합은 seed 결과를 보기 전에 고정했고, XGB 가중치만 0.8/0.9/1.0 중 val BA로 선택 |
| 확률 출력 / 보정 방식 | 헤드의 log-softmax 출력(보정 없음). 계층형 헤드: p(모델) = p(개발사) × p(모델 \| 개발사), 개발사 손실 가중치 0.5. epoch은 val 최고값을 고르지 않고 epoch 3으로 고정 |

## 3. 모델 학습 결과

최종 앙상블 예측(`preds/val_C_FINAL.csv`)을 공통 `eval/evaluate.py`로 계산했다.

### 3.1 개발사 분류 (6 Classes)

| 평가 지표 | Validation | Test |
| --- | --- | --- |
| Accuracy | 0.8384 | 미평가 |
| Precision (Macro) | 0.8243 | 미평가 |
| Recall (Macro) = `company_BA` (확률 합산) | 0.7061 | 미평가 |
| F1-score (Macro) | 0.7424 | 미평가 |
| F1-score (Weighted) | 0.8250 | 미평가 |
| Log loss / ECE | 0.4502 / 0.0447 | 미평가 |

`company_BA_argmax`(1위 모델의 개발사) = 0.7235.

### 3.2 개별 모델 분류 (18 Classes)

| 평가 지표 | Validation | Test |
| --- | --- | --- |
| Accuracy (= BA) | **0.5505** | 미평가 |
| Precision (Macro) | 0.5483 | 미평가 |
| Recall (Macro) | 0.5505 | 미평가 |
| F1-score (Macro) | 0.5433 | 미평가 |
| F1-score (Weighted) | 0.5433 | 미평가 |
| Log loss / ECE | 1.2596 / 0.0587 | 미평가 |
| 형제 정확도 | 0.6349 | 미평가 |

### 3.3 설정별 결과 (val, 18개 모델 BA, epoch 3)

| ID | 변경 사항 | 질문 묶음 | BA | 배정 BA (후처리) | 개발사 BA | train BA | epoch 1 → 2 → 3 |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| Q1 | 동결 본체 + 헤드 | 없음 | 0.237 | 0.457 | 0.467 | 1.000 | – |
| Q1b | Q1 + 질문 중심화 | 중심화 | 0.437 | 0.482 | 0.650 | 미기록 | – |
| Q2 | LoRA (**응답 단독 최종**) | 없음 | **0.328** | 0.455 | 0.554 | 0.866 | 0.212 → 0.331 → 0.328 |
| Q3 | LoRA + 질문 중심화 | 중심화 | 0.482 | 0.604 | 0.682 | 0.984 | 0.404 → 0.492 → 0.482 |
| V1 | Q3 + 규제(label smoothing 0.1, LoRA dropout 0.1, wd 0.05) | 중심화 | 0.472 | 0.535 | 0.635 | 0.993 | 미채택 |
| V2 | 문장 자르기 증강 | 중심화 | 0.470 | 0.561 | 0.671 | 0.950 | 미채택 |
| V3 | Q3 + R-Drop(α 1.0) | 중심화 | 0.492 | 0.606 | 0.686 | 0.985 | 0.422 → 0.457 → 0.492 |
| Q4 | V3 + 계층형 헤드 | 중심화 | 0.500 | 0.636 | 0.715 | 0.977 | 0.394 → 0.523 → 0.500 |

**seed 반복 (BA / 배정 BA)**

| 모델 | seed 42 | seed 43 | seed 44 | 평균 ± 표준편차 (BA) |
| --- | --- | --- | --- | ---: |
| V3 | 0.492 / 0.606 | 0.480 / 0.581 | 0.482 / 0.624 | 0.485 ± 0.007 |
| Q4 | 0.500 / 0.636 | 0.492 / 0.614 | 0.518 / 0.624 | **0.503 ± 0.013** |

- seed 43·44의 V3는 Kaggle Version 2에서 다시 학습됐고 Version 1과 값이 조금 다르다(seed 44: 0.4848 → 0.4823). 위 표와 앙상블은 Version 2 값을 쓴다. GPU 연산이 완전히 결정적이지 않기 때문으로 본다.
- seed 1개에서는 Q4가 V3보다 +0.8%p라 보류했지만, seed 3개 평균에서는 +1.9%p였다.

**앙상블 (확률 평균, val)**

| 조합 | BA | 배정 BA | 개발사 BA | 형제 정확도 | Macro F1 | LogLoss | ECE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| V3 seed 42 | 0.4924 | 0.6061 | 0.6860 | 0.5786 | 0.4865 | 1.5428 | 0.1691 |
| V3 × 3 seed | 0.5126 | 0.6439 | 0.6822 | 0.6027 | 0.5079 | 1.3706 | 0.1178 |
| Q4 × 3 seed | 0.5177 | 0.6515 | 0.7087 | 0.5903 | 0.5083 | 1.3357 | 0.1015 |
| LoRA 3종 평균 (Q3 + V3 + Q4, seed 42) | 0.4949 | 0.6490 | 0.6902 | 0.5700 | – | – | – |
| V3×3 + Q4×3 | 0.5480 | 0.6742 | 0.7045 | 0.6382 | 0.5421 | 1.2883 | 0.0751 |
| **최종: V3×3 + Q4×3 (×0.8) + XGB (×0.2)** | **0.5505** | 0.6717 | 0.7061 | 0.6349 | 0.5433 | 1.2596 | 0.0587 |

- XGB 가중치별 val BA: 0.8 → 0.5505, 0.9 → 0.5455, 1.0 → 0.5480.
- 최종 − V3 seed 42 (질문 단위 paired bootstrap 1,000회): BA +0.058, 95% 신뢰구간 [+0.030, +0.083] · 배정 BA +0.066, [+0.003, +0.129].
- 배정 BA는 XGB 결합 전(0.6742)이 약간 높았지만, 미리 정한 기준(18개 모델 BA)에 따라 최종 조합을 유지했다.

### 3.4 학습 정보

| 항목 | 결과 |
| --- | --- |
| 학습 시간 | R-Drop 설정 1회(3 epoch) 약 66~72분 (T4, epoch당 약 21분, 첫 step에 Triton 컴파일 포함). 예측: 모델 1개당 val+test 약 73초 |
| 최대 GPU 메모리 | 5.4~5.5GB |
| 최종 선택된 모델 | V3(R-Drop) seed 42/43/44 + Q4(R-Drop + 계층형 헤드) seed 42/43/44 확률 평균 × 0.8 + XGBoost × 0.2 |
| 최종 Epoch (해당 시) | 3 (고정) |
| 주요 특이사항 | 저장한 어댑터로 val을 다시 예측해 학습 당시 val 예측과 비교: 8개 중 7개 100% 일치, Q4 seed 43은 99.7%(396개 중 1개 다름, BA 0.4924 → 0.4949). XGB 가중치와 XGB 조기 종료에 val을 써서 val 점수는 약간 낙관적일 수 있다 |

## 4. 모델 분석

### 4.1 Confusion Matrix

**개발사 6개 분류** (최종 앙상블, 행=실제, 열=예측, 확률 합산)

| 실제 \ 예측 | Anthropic | Google | Moonshot | OpenAI | Qwen | ZAI |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Anthropic (110) | **98** | 0 | 0 | 3 | 1 | 8 |
| Google (88) | 2 | **86** | 0 | 0 | 0 | 0 |
| Moonshot (22) | 4 | 0 | **10** | 7 | 0 | 1 |
| OpenAI (110) | 0 | 1 | 0 | **108** | 1 | 0 |
| Qwen (22) | 1 | 1 | 0 | 9 | **11** | 0 |
| ZAI (44) | 21 | 1 | 2 | 1 | 0 | **19** |

- 주요 오분류: **ZAI → Anthropic 21/44**(XGBoost는 27/44), Qwen → OpenAI 9/22, Moonshot → OpenAI 7/22. Google·OpenAI는 거의 다 맞힌다.

**개별 모델 18개 분류**

- 재현율이 낮은 모델: glm-5.3-flash 0.136, glm-5.3 0.227, claude-fable-5 · gpt-5.6-sol · gpt-5.6-terra 0.318. 높은 모델: claude-opus-4-6 · gemini-3.1-pro-preview · gpt-6-astra 0.818.
- 주요 오분류 모델 쌍(개수/22): gpt-5.6-terra → gpt-5.6-luna 7, gpt-5.6-sol → gpt-5.6-terra 6, gemini-3.7-flash → gemini-3.8-flash 6, gemini-3.6-flash → gemini-3.1-pro-preview 6, claude-fable-5 → glm-5.3-flash 6, kimi-k3 → gpt-5.5 5.
- 그래프 파일은 만들지 않았다. 위 수치는 `preds/val_C_FINAL.csv`와 `eval/evaluate.py`로 다시 계산할 수 있다.

### 4.2 주요 특징 분석

| 순위 | 특징 | 중요도/계수 등 | 해석 |
| --- | --- | --- | --- |
| 1 | 질문 중심화 | Q2 → Q3 BA +15.4%p | 같은 질문 응답의 공통 성분(주제)을 빼면 모델별 차이가 드러난다는 가설과 맞다 |
| 2 | LoRA 미세조정 | Q1 → Q2 +9.1%p (중심화 없음), Q1b → Q3 +4.5%p (중심화) | 사전학습 표현을 이 과제 쪽으로 조정 |
| 3 | 앙상블 (seed·구조) | V3 seed 42 → V3×3+Q4×3 +5.6%p | seed 평균 약 +2%p, 구조가 다른 V3·Q4 결합으로 추가 개선 |
| 4 | 계층형 헤드 | seed 3개 평균 +1.9%p, 개발사 BA +2~3%p | 개발사를 먼저 맞히도록 보조 손실을 둠 |
| 5 | R-Drop | Q3 → V3 +1.0%p | 같은 입력의 두 번 예측을 비슷하게 맞추는 규제 |
- **분석 방법:** 한 가지씩 바꾼 설정 비교(ablation). 신경망 내부 특징 중요도·SHAP은 측정하지 않았다.
- **LLM 문체와 관련된 주요 발견:** 개발사 수준은 0.71이지만 18개 모델 수준은 0.55다. 오류 대부분이 같은 개발사 형제 모델 사이에서 생기고, 개발사 사이에서는 ZAI(GLM)가 Anthropic으로 가장 많이 간다.

### 4.3 오분류 사례 분석

| 실제 모델 | 예측 모델 | 오분류 원인 분석 |
| --- | --- | --- |
| gemini-3.6-flash (`5114fcac6158`, p055) | gemini-3.1-pro-preview (확률 0.86) | 같은 개발사 형제 모델. 은유가 많은 서술형 문체가 둘 다 비슷하다는 가설 |
| claude-fable-5 (`c6bc20f1d6ba`, p073) | claude-sonnet-5 (0.82) | Claude 형제 사이 혼동. 개발사 신호는 강하지만 세부 모델 신호가 약함 |
| gemini-3.7-flash (`857db7e94cd3`, p141) | gemini-3.8-flash (0.79) | 연속 버전 모델끼리 혼동(이 쌍은 XGBoost에서도 가장 많이 혼동) |
| kimi-k3 (`8d6fdd5ceb85`, p051) | gpt-5.5 (0.78) | 모델이 1개뿐인 개발사(Moonshot)가 큰 개발사(OpenAI)로 끌려감. 개발사 혼동행렬의 Moonshot → OpenAI 7개와 같은 경향 |
| gemini-3.6-flash (`57869dc05c91`, p022) | gemini-3.1-pro-preview (0.78) | 첫 사례와 같은 쌍. 확신이 높은 오류가 이 쌍에 몰려 있음 |

원인은 관찰에 따른 가설이며 확인 실험은 하지 않았다.

## 5. 심화 모델 추가 실험

### 5.1 적용 기법

- **사전학습 모델:** `Qwen/Qwen3.5-0.8B` (24층: Gated DeltaNet 18 + attention 6, hidden 1024, 약 7.5억 파라미터)
- **적용 기법:** LoRA + 질문 중심화 + R-Drop + 계층형 헤드, seed·모델 앙상블
- **추가 학습 방법:** 본체는 고정하고 LoRA 행렬(0.649%)과 분류 헤드만 학습. 생성 문장 파싱 없이 18개 확률을 직접 출력
- **실험 목적:** 같은 조건에서 동결 본체 대비 LoRA 추가 학습의 효과와 비용 확인(C-4), 작은 데이터에서 과적합 대응 방법 비교

### 5.2 추가 기법 적용 전후 비교

Q1(동결)의 Macro F1은 기록하지 않아 BA로 비교한다(클래스당 표본이 같아 BA = Accuracy = Macro Recall).

| 분류 대상 | 기본 모델 (Q1 동결) | 심화 기법 적용 | 변화량 |
| --- | --- | --- | --- |
| 개발사 6개 (`company_BA`) | 0.467 | Q2 LoRA 0.554 / 최종 0.706 | +0.087 / +0.239 |
| 개별 모델 18개 (BA) | 0.237 | Q2 LoRA 0.328 / 최종 0.551 | +0.091 / +0.314 |
- **추가 기법 적용 효과:** LoRA 단독(응답 단독 예측)으로 +9.1%p, 질문 중심화가 +15.4%p로 가장 크고, R-Drop·계층형 헤드·앙상블이 차례로 더했다. 규제 강화(V1)와 문장 자르기 증강(V2)은 val을 낮춰 채택하지 않았다.
- **주요 발견:** train BA 0.98 대 val 0.50으로 과적합이 남는다. 같은 seed로 다시 학습해도 결과가 달라질 만큼 작은 데이터·작은 val이어서 seed 평균과 질문 단위 신뢰구간으로 판단했다.

## 6. 최종 실험 결과 요약

### 6.1 핵심 결과

- **개발사 분류 Test Macro F1:** 미평가 (val 0.7424)
- **개별 모델 분류 Test Macro F1:** 미평가 (val 0.5433)
- **가장 중요한 문체 특징:** 질문 중심화로 주제 성분을 뺀 임베딩 차이 (4.2절)
- **자주 혼동된 모델:** GPT 5.6 형제(terra·luna·sol), gemini-3.7-flash ↔ 3.8-flash, gemini-3.6-flash → 3.1-pro, GLM ↔ Claude

### 6.2 결론

**실험을 통해 확인한 점**

- 작은 데이터(1,889개)에서도 사전학습 모델 + LoRA는 XGBoost(0.38)보다 높은 0.55를 얻었다.
- 가장 큰 개선은 모델 구조보다 질문 중심화였다.
- 단일 seed 비교는 결론이 바뀔 수 있어, seed 3개 평균과 앙상블로 판단했다.

**모델의 장점**

- 전체의 0.65%만 학습해 T4 한 장(최대 5.5GB)으로 학습할 수 있다.
- 앙상블 후 확률 보정이 좋아졌다(ECE 0.169 → 0.059).

**모델의 한계**

- 과적합(train 0.98 대 val 0.50)이 남고, 규제·증강은 효과가 없었다.
- 질문 중심화는 같은 질문의 응답 18개가 함께 주어질 때만 쓸 수 있다. 응답 단독 예측은 Q2의 0.328이다.
- 학습 비용이 크다(최종 6개 모델 학습에 T4 약 7시간). XGBoost는 CPU 몇 분이다.
- val 22문항이라 1~2%p 차이는 불확실하다. test 점수는 팀 통합 채점 때 확인한다.

---

## 7. 실험 산출물

| 산출물 | 저장 경로 |
| --- | --- |
| 학습 코드 | `seorin`: [`experiments/C_qwen_lora.py`](https://github.com/seoultech-Machine-Learning-26-2/LLM-name-tagging/blob/0434772/experiments/C_qwen_lora.py) · 앙상블·예측: [`experiments/C_ensemble_predict.py`](https://github.com/seoultech-Machine-Learning-26-2/LLM-name-tagging/blob/0434772/experiments/C_ensemble_predict.py) |
| 모델 체크포인트 | 깃허브에 올리지 않음(LoRA 어댑터 약 20MB씩). seed 42: 개인 Google Drive `mlproject/results/` 및 Kaggle 비공개 데이터셋 `yangseorin/c-assets` · seed 43·44: Kaggle 노트북 `yangseorin/llmnametagging` Version 2 Output `results/` |
| 설정 / 체크포인트 (필요 시) | 복원: `AutoModelForCausalLM.from_pretrained('Qwen/Qwen3.5-0.8B').model`에 `PeftModel.from_pretrained(본체, {run}_lora_ep3)`를 얹고 `{run}_head_ep3.pt`를 `FlatHead`/`HierHead`에 불러온다(`C_ensemble_predict.py` 셀 3). 팀원이 쓰려면 Kaggle 데이터셋·노트북 공유가 필요하다 |
| 평가 결과 CSV | `seorin` [`preds/`](https://github.com/seoultech-Machine-Learning-26-2/LLM-name-tagging/tree/5869bb7/preds): `val_C_FINAL.csv`, `test_C_FINAL.csv`, `test_C_{Q2,Q3,V3_seed42~44,Q4_seed42~44}.csv` (test 채점 없음) · 설정별 val 예측은 위 체크포인트 위치에 함께 있음 |
| Confusion Matrix | 4.1절 표 (그림 없음) |
| 특징 분석 그래프 | 없음 (4.2절 표) |
| 최종 실험 보고서 | 이 문서 · 요약: [C 실험 요약](C-summary.md) |
