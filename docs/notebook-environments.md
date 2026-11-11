# Kaggle Notebook·Google Colab 사용 안내

> 상태: 팀 실행 안내 · 최종 확인: 2026-11-11 · B/Codex

## 1. 어떤 환경을 쓸까?

둘 다 브라우저에서 Python 셀을 실행하는 노트북 환경이다. 팀원마다 편한 환경을 써도 입력 분할·코드 버전·모델 설정·평가 방식은 동일하게 맞춘다. 환경이 다르면 GPU·라이브러리 버전과 실행 시간을 함께 기록한다.

| 항목 | Kaggle Notebook | Google Colab |
| --- | --- | --- |
| 시작 | Kaggle 로그인 → Code → New Notebook | Google 로그인 → Colab → 새 노트북 |
| 고전 모델 | CPU로 먼저 실행 | CPU로 먼저 실행 |
| 심화 모델 | 설정에서 사용 가능한 GPU 선택 | 런타임 유형 변경에서 사용 가능한 GPU 선택 |
| 입력 관리 | Dataset 또는 코드 자료를 Add Input으로 연결 | Drive에 팀 입력과 코드 압축파일을 넣고 연결 |
| 결과 보관 | `/kaggle/working`에 출력하고 저장 버전·Output 확인 | Drive 폴더에 출력하거나 런타임 종료 전 다운로드 |
| 공유 | 노트북·Dataset 공개 범위를 팀에 맞게 설정 | 공유 권한을 팀원에게 부여 |

Colab 무료판은 학생이 아니어도 사용 가능하다. 무료 GPU는 무제한·상시 보장이 아니며 할당·한도가 달라진다. 학생용 유료 요금제 혜택은 국가·시기·계정 조건을 따로 확인한다. 대학생이면 자동으로 Colab Pro를 무료로 받는다고 가정하지 않는다. [Colab 공식 FAQ](https://research.google.com/colaboratory/faq.html?hl=ko), [학생 혜택 공식 안내](https://support.google.com/googleone/answer/17422238?hl=ko)

Kaggle의 GPU도 계정에 표시되는 사용 가능 자원·잔여 할당을 확인한다. GPU 옵션이 없으면 계정 인증·플랫폼 안내를 확인하고 CPU에서 고전 모델부터 실행한다. 고정된 무료 시간이나 특정 GPU 제공을 보장하지 않는다. [Kaggle GPU 사용 안내](https://www.kaggle.com/docs/efficient-gpu-usage)

## 2. 공통 준비: 같은 입력·코드 가져오기

1. 팀에서 실행할 Git 커밋을 고정한다. GitHub에서 그 버전의 소스 ZIP을 내려받는다. `data/split/`, `features/`, 평가 코드와 담당 실험 코드가 포함됐는지 확인한다.
2. `.env`, API 키, 개인 파일, `.venv`는 업로드하지 않는다. 데이터와 모델의 사용·공유 조건을 확인하고 필요한 팀원에게만 접근을 준다.
3. 이미 있는 A의 train/val/test를 재사용한다. 원본 CSV로 임의 재분할하지 않는다.
4. 실행에 필요한 패키지를 먼저 확인한다. `requirements-b.txt`는 B 특징 추출 의존성만 포함한다. XGBoost·PyTorch·transformers·PEFT·sentence-transformers 등은 담당 코드에 필요한 버전을 확인해 별도 기록한다. 모든 패키지를 무조건 최신으로 업그레이드하지 않는다.
5. 고전·심화 학습 스크립트는 다음 작업의 산출물이다. 아래 특징 추출 예시만 실행했다고 모델 학습이 완료되는 것은 아니다.

## 3. Kaggle에서 실행

1. 새 노트북을 만들고 팀 코드 ZIP과 입력을 비공개 Dataset 등으로 등록한 뒤 **Add Input**에서 연결한다. ZIP이 자동으로 풀렸는지 Input 파일 목록을 확인한다.
2. GPU가 필요한 심화 실험은 **Settings → Accelerator → GPU**를 선택한다. 모델 다운로드·패키지 설치에 인터넷이 필요하면 해당 노트북의 Internet 설정을 확인한다. 제공되지 않는 옵션은 플랫폼 안내를 따른다.
3. 입력은 읽기 전용 `/kaggle/input` 아래에 있다. 정확한 폴더는 노트북에서 확인한다. 원본에 결과를 쓰지 말고, 코드를 `/kaggle/working`으로 복사하거나 ZIP을 그곳에 푼다.

```python
from pathlib import Path
print(list(Path('/kaggle/input').iterdir()))
```

코드를 풀어 둔 실제 저장소 경로로 이동한 다음 실행한다. 아래 경로는 예시다.

```python
%cd /kaggle/working/LLM-name-tagging
%pip install -r requirements-b.txt
!python -m features.build --data-dir data/split --kind style --output-dir /kaggle/working/results/features
```

4. 담당 학습 코드는 입력·결과 경로를 실제 환경에 맞춰 실행한다. 모델 다운로드는 정확한 체크포인트와 버전을 고정하고 train에만 학습한다.
5. 결과를 `/kaggle/working`에 쓰고 **Save Version**에서 저장한다. **Save & Run All**은 별도 세션에서 처음부터 재실행하므로 시간·GPU 사용이 다시 발생한다. 설정한 저장 방식에서 출력 파일이 실제 보존됐는지 버전의 Output에서 확인하고 CSV·그림·체크포인트를 다운로드한다. [Kaggle 공식 Notebook 안내](https://www.kaggle.com/docs/notebooks)

## 4. Colab에서 실행

1. 새 노트북을 만들고 **런타임 → 런타임 유형 변경**에서 필요한 경우 GPU를 선택한다. 무료 GPU 할당이 안 되면 고전 모델은 CPU로 진행하고 심화 실험의 자원 확보를 먼저 확인한다.
2. 팀 코드 ZIP과 데이터를 Google Drive에 업로드한다. ZIP을 쓴다면 `/content`에 압축 해제하고 코드 폴더로 이동한다. Drive 연결 코드는 다음과 같다.

```python
from google.colab import drive
drive.mount('/content/drive')
```

3. 아래 예시의 저장소·결과 폴더를 본인 Drive 구조에 맞춘다. 공백이 있는 경로는 따옴표로 감싼다.

```python
%cd /content/drive/MyDrive/LLM-name-tagging
%pip install -r requirements-b.txt
!python -m features.build --data-dir data/split --kind style --output-dir /content/drive/MyDrive/LLM-results/B/features
```

4. 심화 학습 중 자주 읽는 파일은 `/content`의 로컬 디스크에서 처리하고, 중간 체크포인트와 최종 결과를 Drive에 복사하면 Drive 입출력 부담을 줄일 수 있다. `/content`는 영구 저장소가 아니다. 세션이 끝나기 전에 결과를 보관한다. [Colab 공식 FAQ](https://research.google.com/colaboratory/faq.html?hl=ko)
5. 노트북을 Drive에 저장하고 **파일 → 다운로드 → .ipynb**로 코드도 확보한다. 노트북에 보이는 그래프만 남기지 말고 원본 CSV·그림 파일·설정을 따로 저장한다. 공유받은 노트북도 실행하면 각자의 런타임에서 실행되므로 파일과 패키지 준비가 필요하다.

## 5. 실행 확인과 결과 보관

```python
import sys, platform, importlib.metadata as md
print(platform.platform(), sys.version)
for name in ['numpy', 'pandas', 'scikit-learn', 'torch', 'transformers',
             'peft', 'sentence-transformers', 'xgboost']:
    try:
        print(name, md.version(name))
    except md.PackageNotFoundError:
        print(name, 'not installed')
```

심화 환경에서 PyTorch가 설치된 뒤 GPU 연결을 확인한다. GPU 메뉴를 선택해도 코드의 모델·배치를 GPU로 보내지 않으면 GPU 학습이 되지 않는다.

```python
import torch
print('CUDA:', torch.cuda.is_available())
if torch.cuda.is_available():
    print(torch.cuda.get_device_name(0))
```

메모리가 부족하면 배치 크기·입력 길이를 줄여 작은 실행부터 확인하고 바뀐 설정을 기록한다. GPU가 두 개 보여도 자동으로 두 개를 모두 쓰는 것은 아니다. 사용하지 않을 때 세션을 종료한다.

**11/18까지 각자 남길 것:** 고전·심화 결과 템플릿, 실제 실행 노트북/스크립트, 코드 커밋·환경 버전·GPU·seed·설정·입력 해시, 예측/평가 CSV, 그림, 필요 체크포인트, 실행 로그와 Notebook 버전 링크. 결과는 플랫폼과 별도로 팀 저장 위치에도 보관한다. 대형 체크포인트는 Git에 직접 넣기보다 접근 가능한 별도 위치와 복원 방법을 문서에 적는다. 시험 전 휴식 기간에 실행을 유지하거나 실험을 이어가는 일정은 배정하지 않는다.
