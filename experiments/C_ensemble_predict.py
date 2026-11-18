# C · 앙상블(val) + test 예측 (학습 없음, test 채점 없음)
#
# - 2026-11-18 Kaggle 노트북에서 GPU T4로 실행한 셀을 그대로 옮겼다 (셀 1~4).
# - Kaggle 입력: `dataset`(train/val), `c-assets`(seed 42 어댑터·헤드·val 예측, evaluate.py, stylometry.py, test.csv),
#   노트북 `llmnametagging` Version 2 출력(seed 43·44 어댑터·헤드·val 예측).
# - 셀 2: XGBoost(full+질문 중심화, seed 42/43/44)를 다시 학습하고, val에서 미리 정한 조합
#   (V3×3 + Q4×3 평균, XGB 가중치 0.8/0.9/1.0 중 val 18개 모델 BA로 선택)을 계산한다.
# - 셀 3: 저장된 어댑터 8개로 val을 다시 예측해 학습 당시 val 예측과 일치하는지 확인한 뒤 test 예측을 저장한다.
#   test.csv의 정답 열(`model`)은 읽자마자 삭제하므로 이 코드는 test 점수를 계산하지 않는다.
# - 출력: preds/val_C_*.csv, preds/test_C_*.csv, preds/test_nolabel.csv (형식: id + CLASSES 18개 확률)

# %% 셀 1: 설치
# !pip uninstall -y -q torchao
# !pip install -q -U transformers peft accelerate flash-linear-attention
import os, glob, time
# !nvidia-smi --query-gpu=name,memory.total --format=csv

# %% 셀 2
# ===== 셀 B: val 앙상블 + XGBoost 결합 =====
import glob, os, re, importlib.util, numpy as np, pandas as pd, xgboost as xgb
from scipy.optimize import linear_sum_assignment
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import TruncatedSVD
IN = '/kaggle/input'

def find(pat):                      # 여러 입력에 같은 파일이 있으면 하나만
    fs = {}
    for f in sorted(glob.glob(f'{IN}/**/{pat}', recursive=True)): fs.setdefault(os.path.basename(f), f)
    return list(fs.values())
def load_py(name, path):
    spec = importlib.util.spec_from_file_location(name, path); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m
E = load_py('team_eval', find('evaluate.py')[0])                 # 공식 평가 코드
SM = load_py('stylometry', find('stylometry.py')[0])             # B 문체 특징
CLS = E.CLASSES; cid = {c: i for i, c in enumerate(CLS)}
D = os.path.dirname(find('val.csv')[0]) + '/'
tr = pd.read_csv(D + 'train.csv'); va = pd.read_csv(D + 'val.csv').sort_values('id').reset_index(drop=True)
ytr, yva = tr.model.map(cid).values, va.model.map(cid).values
assert len(tr) == 1889 and len(va) == 396

def readP(f, ids):
    df = pd.read_csv(f); assert list(df.columns) == ['id'] + CLS, f
    P = df.set_index('id').loc[ids, CLS].to_numpy(float); return P / P.sum(1, keepdims=True)
def group(pat, n=None):
    fs = find(pat); print(f'{pat:22s} → {[os.path.basename(f) for f in fs]}')
    assert fs and (n is None or len(fs) == n), f'{pat}: 파일 {len(fs)}개 (기대 {n})'
    return [readP(f, va.id) for f in fs]

# ---------- XGBoost (full + 중심화, seed 42/43/44 평균) — 고정 설정 ----------
FW = "the a an and or but if of to in on at by for with from as that which who this these those it its is are was were be been being have has had do does not no can could would should may might will shall must very more most also just only even such so than then there here what when where why how all any each every some many much both either neither one two first however therefore thus moreover yet still often perhaps rather quite into onto through across between among within without beyond about".split()
st = SM.StylometricTransformer().fit(tr.text)
cvec = CountVectorizer(vocabulary=FW, token_pattern=r"\b\w+\b")
tf = TfidfVectorizer(analyzer='char', ngram_range=(3, 5), lowercase=False, min_df=2, max_features=30000, sublinear_tf=True).fit(tr.text)
svd = TruncatedSVD(100, random_state=42).fit(tf.transform(tr.text))
def xfeat(df):
    T = df.text
    A = pd.DataFrame(st.transform(T), columns=st.get_feature_names_out())
    n = np.array([max(len(re.findall(r'\b\w+\b', t)), 1) for t in T])[:, None]
    F_ = pd.DataFrame(cvec.transform(T).toarray() / n, columns=['fw_' + w for w in FW])
    S = pd.DataFrame(svd.transform(tf.transform(T)), columns=[f'svd{i}' for i in range(100)])
    X = pd.concat([A, F_, S], axis=1)
    return X - X.groupby(df.prompt_id.values).transform('mean')      # 질문 중심화
Xtr, Xva = xfeat(tr), xfeat(va)
XGB_MODELS = []
for s in [42, 43, 44]:
    m = xgb.XGBClassifier(objective='multi:softprob', n_estimators=3000, max_depth=3, learning_rate=0.05, subsample=0.8,
                          colsample_bytree=0.8, min_child_weight=5, reg_lambda=10, early_stopping_rounds=50,
                          eval_metric='mlogloss', n_jobs=4, tree_method='hist', random_state=s)
    XGB_MODELS.append(m.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)); print('XGB seed', s, 'trees', m.best_iteration)
P_xgb = np.mean([m.predict_proba(Xva) for m in XGB_MODELS], 0).astype(float); P_xgb /= P_xgb.sum(1, keepdims=True)

# ---------- Qwen 예측 파일 ----------
V3 = group('val_V3_rdrop*.csv', 3); Q4 = group('val_Q4*.csv', 3)
Q2 = group('val_Q2*.csv'); Q3 = group('val_Q3*.csv')
avg = lambda L: np.mean(L, 0)
P_qwen = avg(V3 + Q4)                                                  # ★ 최종 Qwen 부분 (미리 고정)

def hung(P, pids):
    pred = P.argmax(1).copy()
    for p in np.unique(pids):
        i = np.where(pids == p)[0]; r, c = linear_sum_assignment(-np.log(P[i] + 1e-9)); pred[i[r]] = c
    return pred
def ba(y, pred): return np.mean([(pred[y == k] == k).mean() for k in np.unique(y)])

# ---------- XGB 가중치: 0.8 / 0.9 / 1.0 중 val model_BA로 선택 (동점이면 큰 값) ----------
W = {w: float(ba(yva, (w * P_qwen + (1 - w) * P_xgb).argmax(1))) for w in [0.8, 0.9, 1.0]}
W_BEST = max(W, key=lambda w: (round(W[w], 6), w)); print('가중치별 val BA:', {k: round(v, 4) for k, v in W.items()}, '→ 선택', W_BEST)
P_final = W_BEST * P_qwen + (1 - W_BEST) * P_xgb

ROWS = {'XGBoost (full+중심화, 3seed)': P_xgb, 'Q2 LoRA': avg(Q2), 'Q3 LoRA+중심화': avg(Q3),
        'V3 R-Drop seed42': V3[0], 'V3 × 3seed': avg(V3), 'Q4 계층형 seed42': Q4[0], 'Q4 × 3seed': avg(Q4),
        'LoRA 3종 평균 (Q3+V3+Q4, seed42)': avg([avg(Q3), V3[0], Q4[0]]), 'V3×3 + Q4×3': P_qwen,
        f'★ FINAL = V3×3+Q4×3 (+XGB, w={W_BEST})': P_final}
def table(rows, y, pids):
    out = []
    for k, P in rows.items():
        s = E.score(y, P)
        out.append({'run': k, 'model_BA': s['model_BA'], '배정_BA': ba(y, hung(P, pids)), 'company_BA': s['company_BA'],
                    'company_argmax': s['company_BA_argmax'], 'sibling': s['sibling_Acc'], 'F1': s['model_F1'],
                    'LogLoss': s['model_LogLoss'], 'ECE': s['model_ECE']})
    return pd.DataFrame(out).set_index('run').round(4)
pd.set_option('display.width', 250); pd.set_option('display.max_colwidth', 45)
VAL_TABLE = table(ROWS, yva, va.prompt_id.values); print(VAL_TABLE.to_string())

# ---------- 질문 단위 paired bootstrap: FINAL vs V3 seed42 ----------
def boot(PA, PB, y, pids, n=1000, seed=0, hun=False):
    rng = np.random.default_rng(seed); u = np.unique(pids)
    pa = hung(PA, pids) if hun else PA.argmax(1); pb = hung(PB, pids) if hun else PB.argmax(1)
    idx = {p: np.where(pids == p)[0] for p in u}; d = []
    for _ in range(n):
        i = np.concatenate([idx[p] for p in rng.choice(u, len(u))]); d.append(ba(y[i], pb[i]) - ba(y[i], pa[i]))
    return np.mean(d), np.percentile(d, 2.5), np.percentile(d, 97.5)
for hun in [False, True]:
    m_, lo, hi = boot(V3[0], P_final, yva, va.prompt_id.values, hun=hun)
    print(f"FINAL − V3 seed42 ({'배정' if hun else '순수'}): {m_:+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]")

# %% 셀 3
# ===== 셀 C: 저장된 모델로 val 재확인 + test 예측 (학습 없음, 채점 없음) =====
import gc, torch, torch.nn as nn, torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
NAME = 'Qwen/Qwen3.5-0.8B'; OUTP = '/kaggle/working/preds/'; os.makedirs(OUTP, exist_ok=True)
tok = AutoTokenizer.from_pretrained(NAME)
if tok.pad_token is None: tok.pad_token = tok.eos_token

# ---- test: 정답 열은 읽자마자 삭제 (글만 사용) ----
TP = find('test*.csv'); assert len(TP) == 1, TP
te = pd.read_csv(TP[0])
te = te.drop(columns=[c for c in ['model'] if c in te.columns]).sort_values('id').reset_index(drop=True)
assert len(te) == 414 and te.prompt_id.value_counts().eq(18).all()
te.to_csv(OUTP + 'test_nolabel.csv', index=False)                 # 팀원 배포용 (정답 없음)

# ---- 학습 때와 똑같은 헤드 정의 ----
COMPS = ['anthropic', 'google', 'moonshot', 'openai', 'qwen', 'zai']
M2C = torch.tensor([COMPS.index({'claude': 'anthropic', 'gemini': 'google', 'gpt': 'openai', 'qwen': 'qwen', 'glm': 'zai',
                                 'kimi': 'moonshot'}[re.match(r'[a-z]+', c).group()]) for c in CLS], device='cuda')
class FlatHead(nn.Module):
    def __init__(s, d): super().__init__(); s.fc = nn.Linear(d, 18)
    def forward(s, h): return F.log_softmax(s.fc(h), -1), None
class HierHead(nn.Module):
    def __init__(s, d): super().__init__(); s.comp = nn.Linear(d, len(COMPS)); s.model = nn.Linear(d, 18)
    def forward(s, h):
        lc = F.log_softmax(s.comp(h), -1); z = s.model(h); lm = torch.empty_like(z)
        for c in range(len(COMPS)):
            idx = (M2C == c).nonzero(as_tuple=True)[0]
            lm[:, idx] = F.log_softmax(z[:, idx], -1) + lc[:, c:c+1]
        return lm, lc

def load_head(path):
    sd = torch.load(path, map_location='cuda')
    if set(sd) == {'weight', 'bias'}: sd = {'fc.' + k: v for k, v in sd.items()}   # Q2·Q3 옛 형식
    hier = 'comp.weight' in sd; d = (sd['model.weight'] if hier else sd['fc.weight']).shape[1]
    head = (HierHead(d) if hier else FlatHead(d)).cuda(); head.load_state_dict(sd); head.eval()
    return head, d == 2048, hier                                   # 2048차원이면 중심화 사용

@torch.no_grad()
def qpredict(pm, head, center, df):
    out = {}
    for _, g in df.groupby('prompt_id'):
        enc = tok(list(g.text), padding=True, truncation=True, max_length=384, pad_to_multiple_of=64, return_tensors='pt').to('cuda')
        with torch.autocast('cuda', dtype=torch.float16):
            h = pm(**enc).last_hidden_state.float()
        m = enc.attention_mask.unsqueeze(-1).float(); H = (h * m).sum(1) / m.sum(1)
        X = torch.cat([H, H - H.mean(0, keepdim=True)], -1) if center else H
        for i, p in zip(g.index, head(X)[0].exp().cpu().numpy()): out[i] = p
    return np.stack([out[i] for i in df.index])

def save_pred(P, df, fname):
    o = pd.DataFrame(P, columns=CLS); o.insert(0, 'id', df.id.values); o.to_csv(OUTP + fname, index=False)

CFGS = glob.glob(f'{IN}/**/adapter_config.json', recursive=True)
def adir_of(key):                                                  # zip 폴더 이름에 날짜가 붙어도 찾기
    return sorted({os.path.dirname(c) for c in CFGS if re.search(key + r'(?![\w])', c.replace(os.sep, '/'))})
RUNS = [('Q2', r'Q2_\w*_lora_ep3', 'Q2*_head_ep3.pt', 'val_Q2*.csv'), ('Q3', r'Q3_\w*_lora_ep3', 'Q3*_head_ep3.pt', 'val_Q3*.csv'),
        ('V3_seed42', r'V3_rdrop_lora_ep3', 'V3_rdrop_head_ep3.pt', 'val_V3_rdrop_seed42.csv'),
        ('V3_seed43', r'V3_rdrop_seed43_lora_ep3', 'V3_rdrop_seed43_head_ep3.pt', 'val_V3_rdrop_seed43.csv'),
        ('V3_seed44', r'V3_rdrop_seed44_lora_ep3', 'V3_rdrop_seed44_head_ep3.pt', 'val_V3_rdrop_seed44.csv'),
        ('Q4_seed42', r'Q4_hier_lora_ep3', 'Q4_hier_head_ep3.pt', 'val_Q4_hier_seed42.csv'),
        ('Q4_seed43', r'Q4_hier_seed43_lora_ep3', 'Q4_hier_seed43_head_ep3.pt', 'val_Q4_hier_seed43.csv'),
        ('Q4_seed44', r'Q4_hier_seed44_lora_ep3', 'Q4_hier_seed44_head_ep3.pt', 'val_Q4_hier_seed44.csv')]
for name, akey, hpat, vpat in RUNS:                                # 먼저 파일이 다 있는지 확인
    assert len(adir_of(akey)) == 1 and len(find(hpat)) == 1 and len(find(vpat)) == 1, \
        f'{name}: 파일 확인 필요 {adir_of(akey)} {find(hpat)} {find(vpat)}'
print('모델 파일 8개 확인 ✅')

TEST, CHECK = {}, []
for name, akey, hpat, vpat in RUNS:
    t0 = time.time(); adir = adir_of(akey)[0]
    head, center, hier = load_head(find(hpat)[0])
    base = AutoModelForCausalLM.from_pretrained(NAME, dtype=torch.float32).model.cuda()
    pm = PeftModel.from_pretrained(base, adir).eval()
    Pv = qpredict(pm, head, center, va); Pold = readP(find(vpat)[0], va.id)
    same = (Pv.argmax(1) == Pold.argmax(1)).mean()                 # 학습 때 val 예측과 같은지
    CHECK.append(dict(run=name, center=center, hier=hier, val_BA_now=round(ba(yva, Pv.argmax(1)), 4),
                      val_BA_saved=round(ba(yva, Pold.argmax(1)), 4), same_pred=round(same, 3)))
    TEST[name] = qpredict(pm, head, center, te); save_pred(TEST[name], te, f'test_C_{name}.csv')
    print(CHECK[-1], f'{time.time() - t0:.0f}s', '✅' if same > 0.95 else '⚠️ 불일치! 멈추고 알려줘', flush=True)
    del pm, base, head; gc.collect(); torch.cuda.empty_cache()

# ---- XGBoost test 예측 + 최종 앙상블 (조합·가중치는 val에서 이미 고정) ----
P_te_xgb = np.mean([m.predict_proba(xfeat(te)) for m in XGB_MODELS], 0).astype(float); P_te_xgb /= P_te_xgb.sum(1, keepdims=True)
P_te_qwen = np.mean([TEST[k] for k in ['V3_seed42', 'V3_seed43', 'V3_seed44', 'Q4_seed42', 'Q4_seed43', 'Q4_seed44']], 0)
P_te_final = W_BEST * P_te_qwen + (1 - W_BEST) * P_te_xgb
save_pred(P_te_xgb, te, 'test_C_XGB.csv'); save_pred(P_te_final, te, 'test_C_FINAL.csv')
save_pred(P_xgb, va, 'val_C_XGB.csv'); save_pred(P_final, va, 'val_C_FINAL.csv')
print(pd.DataFrame(CHECK).to_string(index=False)); print('저장된 파일:', sorted(os.listdir(OUTP)))

# %% 셀 4: 결과 내려받기
# !cd /kaggle/working && zip -qr c_preds.zip preds && ls -la c_preds.zip
# from IPython.display import FileLink; FileLink('c_preds.zip')
