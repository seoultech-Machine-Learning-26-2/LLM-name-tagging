"""
Name That LLM 공통 평가 코드

- 모든 모델의 예측 파일을 같은 기준으로 검사하고 평가한다
- 예측 파일 형식: id + 18개 확률 (열 순서는 CLASSES)
- 사용법은 eval/README.md 참고
"""

import numpy as np, pandas as pd
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score, log_loss,
                             confusion_matrix, precision_score, recall_score)


# =========================================================
# 1. 라벨 정의
# =========================================================

CLASSES = ['claude-fable-5','claude-opus-4-6','claude-opus-4-8','claude-opus-5','claude-sonnet-5',
           'gemini-3.1-pro-preview','gemini-3.6-flash','gemini-3.7-flash','gemini-3.8-flash',
           'glm-5.3','glm-5.3-flash','gpt-5.5','gpt-5.6-luna','gpt-5.6-sol','gpt-5.6-terra',
           'gpt-6-astra','kimi-k3','qwen3.8-max']
PREFIX = {'claude':'Anthropic','gemini':'Google','glm':'ZAI','gpt':'OpenAI','kimi':'Moonshot','qwen':'Qwen'}
COMPANIES = sorted(PREFIX.values())
M2C = np.array([COMPANIES.index(PREFIX[m.split('-')[0].rstrip('0123456789.')]) for m in CLASSES])
MULTI = np.bincount(M2C, minlength=6) > 1   # 모델이 2개 이상인 회사


# =========================================================
# 2. 지표 계산
# =========================================================

def to_company(P):
    """18개 확률을 회사별로 합쳐 6개 확률로 변환"""
    Q = np.zeros((len(P), 6))
    for m, c in enumerate(M2C):
        Q[:, c] += P[:, m]
    return Q

def ece(y, P, n_bins=15):
    """예측 확신도와 실제 정답률의 차이"""
    conf, hit = P.max(1), P.argmax(1) == y
    bins = np.minimum((conf * n_bins).astype(int), n_bins - 1)
    return float(sum((bins == b).mean() * abs(hit[bins == b].mean() - conf[bins == b].mean())
                     for b in range(n_bins) if (bins == b).any()))

def score(y, P):
    """y: 정답 인덱스(0~17), P: (n, 18) 확률"""
    pred, yc, Pc = P.argmax(1), M2C[y], to_company(P)
    predc = Pc.argmax(1)
    # 형제 정확도: 회사를 맞힌 글 중 모델까지 맞힌 비율 (모델이 1개뿐인 회사 제외)
    mask = MULTI[yc] & (M2C[pred] == yc)
    return {
        'model_BA': balanced_accuracy_score(y, pred),
        'model_Precision': precision_score(y, pred, average='macro', labels=range(18), zero_division=0),
        'model_Recall': recall_score(y, pred, average='macro', labels=range(18), zero_division=0),
        'model_F1': f1_score(y, pred, average='macro', labels=range(18), zero_division=0),
        'model_F1w': f1_score(y, pred, average='weighted', labels=range(18), zero_division=0),
        'model_Acc': accuracy_score(y, pred),
        'model_LogLoss': log_loss(y, P, labels=range(18)),
        'model_ECE': ece(y, P),
        'sibling_Acc': (pred[mask] == y[mask]).mean() if mask.any() else np.nan,
        'company_BA': balanced_accuracy_score(yc, predc),
        'company_BA_argmax': balanced_accuracy_score(yc, M2C[pred]),  # 18개 중 1위 모델의 회사로 예측
        'company_Precision': precision_score(yc, predc, average='macro', labels=range(6), zero_division=0),
        'company_Recall': recall_score(yc, predc, average='macro', labels=range(6), zero_division=0),
        'company_F1': f1_score(yc, predc, average='macro', labels=range(6), zero_division=0),
        'company_F1w': f1_score(yc, predc, average='weighted', labels=range(6), zero_division=0),
        'company_Acc': accuracy_score(yc, predc),
        'company_LogLoss': log_loss(yc, Pc, labels=range(6)),
        'company_ECE': ece(yc, Pc),
    }


# =========================================================
# 3. 데이터·예측 파일 불러오기
# =========================================================

def load_data(train_path, val_path, test_path):
    """A의 CSV 3개를 합쳐 정답표를 만든다"""
    parts = {'train': train_path, 'val': val_path, 'test': test_path}
    data = pd.concat([pd.read_csv(p).assign(split=k) for k, p in parts.items()], ignore_index=True)
    n = data.split.value_counts().to_dict()
    assert n == {'train': 1889, 'test': 414, 'val': 396}, f'A의 분할 파일이 아님: {n}'
    assert data.id.is_unique and data.groupby('prompt_id').split.nunique().eq(1).all(), '분할 오류'
    return data[['id', 'prompt_id', 'model', 'split']]

def load_run(paths, data, split='val', open_test=False):
    """예측 파일을 검사하고 불러온다. 시드가 여러 개면 파일 경로를 리스트로 전달"""
    if split not in ('val', 'test'):
        raise ValueError("split은 'val' 또는 'test'만 가능")
    if split == 'test' and not open_test:
        raise PermissionError('test는 정해진 시점에만 open_test=True로 연다')
    paths = [paths] if isinstance(paths, str) else list(paths)
    meta = data[data.split == split].sort_values('id').reset_index(drop=True)
    Ps = []
    for p in paths:
        df = pd.read_csv(p)
        assert list(df.columns) == ['id'] + CLASSES, f'{p}: 컬럼은 id + CLASSES(알파벳순)'
        assert df.id.is_unique and set(df.id) == set(meta.id), f'{p}: id가 평가 대상과 불일치'
        P = df.set_index('id').loc[meta.id, CLASSES].to_numpy(float)
        assert np.isfinite(P).all() and (P >= 0).all(), f'{p}: NaN 또는 음수 확률'
        assert np.allclose(P.sum(1), 1, atol=1e-3), f'{p}: 행 합이 1이 아님'
        Ps.append(P / P.sum(1, keepdims=True))
    y = meta.model.map({m: i for i, m in enumerate(CLASSES)}).to_numpy()
    return {'split': split, 'meta': meta, 'P': np.stack(Ps), 'y': y}


# =========================================================
# 4. 평가
# =========================================================

def evaluate(run):
    """시드 평균 ± 표준편차 (시드 1개면 std는 0)"""
    rows = pd.DataFrame([score(run['y'], P) for P in run['P']])
    return pd.DataFrame({'mean': rows.mean(), 'std': rows.std(ddof=1) if len(rows) > 1 else 0.0})

def _boot(meta, n_boot, seed):
    """질문 단위 복원추출: 같은 질문의 답은 함께 뽑힌다"""
    rng = np.random.default_rng(seed)
    groups = list(meta.groupby('prompt_id').indices.values())
    for _ in range(n_boot):
        yield np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))])

def _ba(y, pred, n):
    tot = np.bincount(y, minlength=n)
    hit = np.bincount(y, weights=pred == y, minlength=n)
    return (hit[tot > 0] / tot[tot > 0]).mean()

def _fast(y, Ps):
    return np.mean([[_ba(y, P.argmax(1), 18), _ba(M2C[y], to_company(P).argmax(1), 6)] for P in Ps], 0)

def bootstrap_ci(run, n_boot=1000, seed=0):
    """BA의 95% 신뢰구간"""
    b = np.array([_fast(run['y'][i], run['P'][:, i]) for i in _boot(run['meta'], n_boot, seed)])
    lo, hi = np.percentile(b, [2.5, 97.5], axis=0)
    return pd.DataFrame({'ci_low': lo, 'ci_high': hi}, index=['model_BA', 'company_BA'])

def compare(base, new, n_boot=1000, seed=0):
    """new - base의 paired bootstrap. ci_low > 0이면 개선"""
    assert base['split'] == new['split'] and base['meta'].id.equals(new['meta'].id), '평가 대상이 다름'
    d = np.array([_fast(base['y'][i], new['P'][:, i]) - _fast(base['y'][i], base['P'][:, i])
                  for i in _boot(base['meta'], n_boot, seed)])
    lo, hi = np.percentile(d, [2.5, 97.5], axis=0)
    out = pd.DataFrame({'diff_mean': d.mean(0), 'ci_low': lo, 'ci_high': hi}, index=['model_BA', 'company_BA'])
    out['improved'] = out.ci_low > 0
    out['worse'] = out.ci_high < 0
    return out

def confusion(run, level='model'):
    """실제 라벨(행) 기준으로 정규화한 혼동행렬. 대각선이 클래스별 recall"""
    y, names = (run['y'], CLASSES) if level == 'model' else (M2C[run['y']], COMPANIES)
    pred = (lambda P: P.argmax(1)) if level == 'model' else (lambda P: to_company(P).argmax(1))
    Ms = [confusion_matrix(y, pred(P), labels=range(len(names)), normalize='true') for P in run['P']]
    return pd.DataFrame(np.mean(Ms, 0), index=names, columns=names)


# =========================================================
# 5. 보고서용 표
# =========================================================

def leaderboard(runs):
    """{'모델명': run} -> 보고서용 비교표"""
    return pd.DataFrame({k: evaluate(r).apply(lambda s: f"{s['mean']:.4f} ± {s['std']:.4f}", axis=1)
                         for k, r in runs.items()}).T

def template_table(val_run, test_run, level='model'):
    """결과 템플릿 3.1(level='company'), 3.2(level='model') 표 생성"""
    rows = {'Accuracy': 'Acc', 'Precision (Macro)': 'Precision', 'Recall (Macro) = BA': 'Recall',
            'F1-score (Macro)': 'F1', 'F1-score (Weighted)': 'F1w', 'Log loss': 'LogLoss', 'ECE': 'ECE'}
    if level == 'model':
        rows['형제 정확도'] = None
    dv, te = evaluate(val_run), evaluate(test_run)
    key = lambda v: 'sibling_Acc' if v is None else f'{level}_{v}'
    return pd.DataFrame({'Validation': [f"{dv.loc[key(v), 'mean']:.4f} ± {dv.loc[key(v), 'std']:.4f}" for v in rows.values()],
                         'Test': [f"{te.loc[key(v), 'mean']:.4f}" for v in rows.values()]}, index=list(rows))


# =========================================================
# 6. 예측 파일 저장
# =========================================================

def save_pred(ids, P, labels, name):
    """ids: 글 id, P: (n, 18) 확률, labels: P의 열 순서 (sklearn은 model.classes_)"""
    df = pd.DataFrame(P, columns=list(labels))[CLASSES]
    df.insert(0, 'id', list(ids))
    df.to_csv(name, index=False)
