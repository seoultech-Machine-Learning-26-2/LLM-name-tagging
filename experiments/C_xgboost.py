# C · XGBoost 탐색 실험 (val 비교, test 미사용)
#
# 실행: 저장소 루트에서 `python experiments/C_xgboost.py` (CPU, 수 분)
# - 2026-11-15~16에 실행한 탐색 스크립트에서 데이터·평가 코드 경로만 저장소 기준으로 바꿨다.
# - 그룹 1: B 문체 32개, 깊이 3/6 × 학습률 0.05/0.1
# - 그룹 2: 깊이 3 + 규제(min_child_weight 5, reg_lambda 10), seed 42/43/44
# - 그룹 3: + 기능어 99개 / + 문자 3~5그램 TF-IDF→SVD 100 / 둘 다(seed 3개)
# - 그룹 4: 전체 특징 + 질문 중심화(같은 prompt_id 18개 평균 빼기, seed 3개) → 최종 설정
#   (질문 중심화는 같은 질문의 응답 묶음을 이용하므로 응답 단독 예측과 구분해 보고한다.)
# - 조기 종료는 val mlogloss 기준(50회). 변환기(TF-IDF·SVD·문체)는 train에만 fit.

import os, sys, re, time, numpy as np, pandas as pd, xgboost as xgb
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'eval')); sys.path.insert(1, ROOT)
import evaluate as E                                   # eval/evaluate.py (공통 평가 코드)
from features.stylometry import StylometricTransformer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import TruncatedSVD
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import balanced_accuracy_score
S = os.path.join(ROOT, 'data', 'split')
tr = pd.read_csv(os.path.join(S, 'train.csv')); va = pd.read_csv(os.path.join(S, 'val.csv'))   # test는 읽지 않는다
cid = {c: i for i, c in enumerate(E.CLASSES)}
ytr = tr.model.map(cid).values; yva = va.model.map(cid).values
st = StylometricTransformer().fit(tr.text)
Xtr = pd.DataFrame(st.transform(tr.text), columns=st.get_feature_names_out()); Xva = pd.DataFrame(st.transform(va.text), columns=Xtr.columns)
print('constant in train:', [c for c in Xtr if Xtr[c].nunique() == 1])
FW = "the a an and or but if of to in on at by for with from as that which who this these those it its is are was were be been being have has had do does not no can could would should may might will shall must very more most also just only even such so than then there here what when where why how all any each every some many much both either neither one two first however therefore thus moreover yet still often perhaps rather quite into onto through across between among within without beyond about".split()
cv = CountVectorizer(vocabulary=FW, token_pattern=r"\b\w+\b")
def fw(T):
    C = cv.transform(T).toarray().astype(float); n = np.array([max(len(re.findall(r'\b\w+\b', t)), 1) for t in T])[:, None]
    return pd.DataFrame(C / n, columns=['fw_' + w for w in FW])
tf = TfidfVectorizer(analyzer="char", ngram_range=(3, 5), lowercase=False, min_df=2, max_features=30000, sublinear_tf=True)  # B build.py 설정
Ttr = tf.fit_transform(tr.text); Tva = tf.transform(va.text)
svd = TruncatedSVD(100, random_state=42).fit(Ttr); sc = [f'svd{i}' for i in range(100)]
Str = pd.DataFrame(svd.transform(Ttr), columns=sc); Sva = pd.DataFrame(svd.transform(Tva), columns=sc)
def center(X, df): return X - X.groupby(df.prompt_id.values).transform('mean')
def hung(P, pids):
    pred = P.argmax(1).copy()
    for p in np.unique(pids):
        idx = np.where(pids == p)[0]; r, c = linear_sum_assignment(-np.log(P[idx] + 1e-9)); pred[idx[r]] = c
    return pred
base = dict(objective='multi:softprob', n_estimators=3000, subsample=0.8, colsample_bytree=0.8, early_stopping_rounds=50,
            eval_metric='mlogloss', n_jobs=2, tree_method='hist')
rows = []
def run(grp, name, A, B, seeds=(42,), **kw):
    Ps = []; t0 = time.time()
    for s in seeds:
        m = xgb.XGBClassifier(**{**base, 'max_depth': 3, 'learning_rate': 0.05, 'random_state': s, **kw}).fit(A, ytr, eval_set=[(B, yva)], verbose=False)
        Ps.append(m.predict_proba(B)); trBA = balanced_accuracy_score(ytr, m.predict_proba(A).argmax(1)); trees = m.best_iteration
    sc_ = [E.score(yva, P) for P in Ps]; mean = lambda k: np.mean([d[k] for d in sc_])
    rows.append(dict(group=grp, run=name, n_feat=A.shape[1], model_BA=mean('model_BA'), sd=np.std([d['model_BA'] for d in sc_]),
                     model_F1=mean('model_F1'), LogLoss=mean('model_LogLoss'), ECE=mean('model_ECE'), sibling=mean('sibling_Acc'),
                     company_BA=mean('company_BA'), hung_BA=np.mean([balanced_accuracy_score(yva, hung(P, va.prompt_id.values)) for P in Ps]),
                     train_BA=trBA, trees=trees, sec=round(time.time() - t0, 1)))
    print(rows[-1]['run'], round(rows[-1]['model_BA'], 3), flush=True)
REG = dict(min_child_weight=5, reg_lambda=10)
for d in [3, 6]:
    for lr in [0.05, 0.1]: run('1_common_B32', f'd{d}_lr{lr}', Xtr, Xva, max_depth=d, learning_rate=lr)
run('2_reg', 'd3+reg', Xtr, Xva, seeds=(42, 43, 44), **REG)
A3 = pd.concat([Xtr, Ftr := fw(tr.text), Str], axis=1); B3 = pd.concat([Xva, Fva := fw(va.text), Sva], axis=1)
run('3_feat', 'B32+funcword', pd.concat([Xtr, Ftr], axis=1), pd.concat([Xva, Fva], axis=1), **REG)
run('3_feat', 'B32+charSVD100', pd.concat([Xtr, Str], axis=1), pd.concat([Xva, Sva], axis=1), **REG)
run('3_feat', 'B32+funcword+charSVD100', A3, B3, seeds=(42, 43, 44), **REG)
run('4_struct', 'full+centered', center(A3, tr), center(B3, va), seeds=(42, 43, 44), **REG)
pd.set_option('display.width', 250)
print(pd.DataFrame(rows).round(3).to_string(index=False))
