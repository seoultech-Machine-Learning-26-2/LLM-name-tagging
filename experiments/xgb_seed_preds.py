# C · XGBoost(full + 질문 중심화) seed별 val·test 예측 저장 (Q-10 결정 4: seed별 파일)
# 학습 설정은 notebooks/kaggle_session_2026-11-18 셀 B의 XGBoost 부분과 같다. 바꾼 것: 입력 경로, seed별 저장, 기존 평균 파일과 비교.
# test는 정답 열이 없는 preds/test_nolabel.csv의 글만 사용한다 (채점 없음).
# 실행: 저장소 루트에서 python experiments/xgb_seed_preds.py
import re, sys, importlib.util, numpy as np, pandas as pd, xgboost as xgb
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import TruncatedSVD

def load_py(name, path):
    spec = importlib.util.spec_from_file_location(name, path); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m
E = load_py('team_eval', 'eval/evaluate.py')                       # 공식 평가 코드
SM = load_py('stylometry', 'features/stylometry.py')                # B 문체 특징
CLS = E.CLASSES; cid = {c: i for i, c in enumerate(CLS)}
tr = pd.read_csv('data/split/train.csv'); va = pd.read_csv('data/split/val.csv').sort_values('id').reset_index(drop=True)
te = pd.read_csv('preds/test_nolabel.csv').sort_values('id').reset_index(drop=True)
assert 'model' not in te.columns and len(te) == 414
ytr, yva = tr.model.map(cid).values, va.model.map(cid).values
assert len(tr) == 1889 and len(va) == 396

# ---------- 셀 B와 같은 특징·설정 ----------
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
Xtr, Xva, Xte = xfeat(tr), xfeat(va), xfeat(te)

def save_pred(P, df, fname):
    o = pd.DataFrame(P, columns=CLS); o.insert(0, 'id', df.id.values); o.to_csv('preds/' + fname, index=False)
def norm(P): P = P.astype(float); return P / P.sum(1, keepdims=True)

PV, PT = [], []
for s in [42, 43, 44]:
    m = xgb.XGBClassifier(objective='multi:softprob', n_estimators=3000, max_depth=3, learning_rate=0.05, subsample=0.8,
                          colsample_bytree=0.8, min_child_weight=5, reg_lambda=10, early_stopping_rounds=50,
                          eval_metric='mlogloss', n_jobs=4, tree_method='hist', random_state=s)
    m.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
    pv, pt = norm(m.predict_proba(Xva)), norm(m.predict_proba(Xte)); PV.append(pv); PT.append(pt)
    save_pred(pv, va, f'val_C_XGB_seed{s}.csv'); save_pred(pt, te, f'test_C_XGB_seed{s}.csv')
    print(f'seed {s}: trees {m.best_iteration}, val model_BA {E.score(yva, pv)["model_BA"]:.4f}', flush=True)

# ---------- 기존 3 seed 평균 파일과 비교 (셀 B와 같은 방식: 평균 후 정규화) ----------
for name, Ps, df in [('val_C_XGB.csv', PV, va), ('test_C_XGB.csv', PT, te)]:
    old = pd.read_csv('preds/' + name).set_index('id').loc[df.id, CLS].to_numpy(float)
    new = norm(np.mean(Ps, 0))
    print(f'{name}: 최대 확률 차이 {np.abs(new - old).max():.2e}, argmax 일치 {(new.argmax(1) == old.argmax(1)).mean():.4f}')
s = E.score(yva, norm(np.mean(PV, 0))); print(f'3 seed 평균 val model_BA {s["model_BA"]:.4f} (문서 0.3813)')
print('seed별 val model_BA 평균 ± sd:', np.round(np.mean([E.score(yva, p)['model_BA'] for p in PV]), 4),
      np.round(np.std([E.score(yva, p)['model_BA'] for p in PV]), 4))
