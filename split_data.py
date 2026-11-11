"""LLM 작성 모델 분류 프로젝트: 데이터 검사 및 Train/Val/Test 분할.

사용 방법 (Colab에서 저장소를 clone한 뒤):
    %run split_data.py

입력: data/train.csv (원본)
출력: data/split/train.csv, val.csv, test.csv, label_map.json
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# ==========================================
# 1. 기본 설정
# ==========================================

# 이 스크립트가 있는 폴더(저장소 최상위)를 기준으로 경로 설정
ROOT = Path(__file__).resolve().parent

# 원본 데이터셋 경로
INPUT_FILE = ROOT / "data" / "train.csv"

# 분할된 파일을 저장할 폴더 (원본 파일 덮어쓰기 방지)
OUTPUT_DIR = ROOT / "data" / "split"

# 실행할 때마다 동일한 분할 결과가 나오도록 시드 고정
SEED = 42

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ==========================================
# 2. 원본 데이터 불러오기
# ==========================================

df = pd.read_csv(INPUT_FILE)

# 필수 컬럼 존재 여부 확인
required = {"id", "prompt_id", "text", "model"}
missing = required - set(df.columns)
if missing:
    raise ValueError(f"필수 컬럼 누락: {sorted(missing)}")

print("=== 데이터셋 기본 정보 ===")
print(f"전체 데이터 수: {len(df)}")
print(f"프롬프트 수: {df['prompt_id'].nunique()}")
print(f"LLM 모델 수: {df['model'].nunique()}")
print(f"컬럼 목록: {list(df.columns)}")


# ==========================================
# 3. 데이터 품질 검사
# ==========================================

print("\n=== 결측치 검사 ===")
print(df.isnull().sum())

print("\n=== 빈 텍스트 검사 ===")
empty_text = df["text"].astype(str).str.strip().eq("").sum()
print(f"빈 텍스트(공백만 있는 경우 포함): {empty_text}")

print("\n=== 중복 검사 ===")
print(f"완전 중복 행: {df.duplicated().sum()}")
print(f"중복 ID: {df['id'].duplicated().sum()}")
print(f"중복 텍스트: {df['text'].duplicated().sum()}")

print("\n=== 원본 모델별 데이터 수 ===")
print(df["model"].value_counts().sort_index())

# 모든 모델의 응답이 다 있지 않은 프롬프트 확인
n_models = df["model"].nunique()
per_prompt = df.groupby("prompt_id")["model"].nunique()
incomplete = per_prompt[per_prompt < n_models]
print(f"\n=== 응답이 빠진 프롬프트 ({len(incomplete)}개) ===")
for pid in incomplete.index:
    have = set(df.loc[df["prompt_id"] == pid, "model"])
    lack = sorted(set(df["model"].unique()) - have)
    print(f"prompt_id={pid}: 빠진 모델 {lack}")

# 결측치, 빈 텍스트, 중복 ID가 발견되면 임의 삭제 없이 실행 중단
if df[list(required)].isnull().any().any():
    raise ValueError("필수 컬럼에 결측치가 있습니다. 확인 후 다시 실행하세요.")
if empty_text > 0:
    raise ValueError("빈 텍스트가 있습니다. 확인 후 다시 실행하세요.")
if not df["id"].is_unique:
    raise ValueError("중복 ID가 존재합니다. 확인 후 다시 실행하세요.")


# ==========================================
# 4. prompt_id 기준 데이터 분할
# ==========================================

# 같은 프롬프트에 대한 모든 LLM 응답은 동일한 데이터셋에 배치
prompt_ids = np.sort(df["prompt_id"].unique())

# 프롬프트 기준 Train 70% / 임시 데이터 30%
train_ids, temp_ids = train_test_split(
    prompt_ids,
    test_size=0.30,
    random_state=SEED,
    shuffle=True,
)

# 임시 데이터 30%를 절반으로 나눠 Val 15% / Test 15%
val_ids, test_ids = train_test_split(
    temp_ids,
    test_size=0.50,
    random_state=SEED,
    shuffle=True,
)

# 각 프롬프트에 속한 모든 행 추출
train_df = df[df["prompt_id"].isin(train_ids)].copy()
val_df = df[df["prompt_id"].isin(val_ids)].copy()
test_df = df[df["prompt_id"].isin(test_ids)].copy()


# ==========================================
# 5. 분할 결과 검증
# ==========================================

# 같은 프롬프트가 두 세트에 동시에 들어가지 않았는지 확인 (데이터 누수 방지)
assert set(train_ids).isdisjoint(val_ids), "train/val 프롬프트 중복"
assert set(train_ids).isdisjoint(test_ids), "train/test 프롬프트 중복"
assert set(val_ids).isdisjoint(test_ids), "val/test 프롬프트 중복"

# 빠진 행 없이 전부 나뉘었는지 확인
assert len(train_df) + len(val_df) + len(test_df) == len(df), "행 수 불일치"

print("\n=== 분할 결과 ===")
for name, part, ids in [
    ("train", train_df, train_ids),
    ("val", val_df, val_ids),
    ("test", test_df, test_ids),
]:
    print(f"{name:5s}: 프롬프트 {len(ids):3d}개 / 행 {len(part):4d}개 ({len(part) / len(df):.1%})")

print("\n=== 세트별 모델 분포 ===")
dist = pd.DataFrame({
    "train": train_df["model"].value_counts(),
    "val": val_df["model"].value_counts(),
    "test": test_df["model"].value_counts(),
}).fillna(0).astype(int).sort_index()
print(dist)


# ==========================================
# 6. 저장
# ==========================================

train_df.to_csv(OUTPUT_DIR / "train.csv", index=False)
val_df.to_csv(OUTPUT_DIR / "val.csv", index=False)
test_df.to_csv(OUTPUT_DIR / "test.csv", index=False)

# 모델 이름 -> 숫자 라벨 (알파벳 순). 팀 전체가 같은 순서를 쓰도록 공유
label_map = {name: i for i, name in enumerate(sorted(df["model"].unique()))}
with open(OUTPUT_DIR / "label_map.json", "w", encoding="utf-8") as f:
    json.dump(label_map, f, ensure_ascii=False, indent=2)

print(f"\n저장 완료: {OUTPUT_DIR}")
print(" - train.csv, val.csv, test.csv, label_map.json")
