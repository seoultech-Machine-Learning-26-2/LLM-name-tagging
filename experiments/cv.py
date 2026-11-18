import re, numpy as np, pandas as pd, xgboost as xgb
from bfeat import B32
from sklearn.model_selection import GroupKFold
from sklearn.metrics import balanced_accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import TruncatedSVD
from scipy.optimize import linear_sum_assignment
tr=pd.read_csv('/mnt/user-data/uploads/6ddfa492-train.csv')
classes=sorted(tr.model.unique()); y=tr.model.map({c:i for i,c in enumerate(classes)}).values
FW="the a an and or but if of to in on at by for with from as that which who this these those it its is are was were be been being have has had do does not no can could would should may might will shall must very more most also just only even such so than then there here what when where why how all any each every some many much both either neither one two first however therefore thus moreover yet still often perhaps rather quite into onto through across between among within without beyond about".split()
cv=CountVectorizer(vocabulary=FW,token_pattern=r"\b\w+\b")
def fw(T):
    C=cv.transform(T).toarray().astype(float); n=np.array([max(len(re.findall(r'\b\w+\b',t)),1) for t in T])[:,None]; return pd.DataFrame(C/n,columns=['fw_'+w for w in FW])
def center(X,pid): return X-X.groupby(pid).transform('mean')
def hung(P,pid):
    pred=P.argmax(1).copy()
    for p in np.unique(pid):
        idx=np.where(pid==p)[0]; r,c=linear_sum_assignment(-np.log(P[idx]+1e-9)); pred[idx[r]]=c
    return pred
P0=dict(objective='multi:softprob',n_estimators=300,max_depth=3,learning_rate=0.05,subsample=0.8,colsample_bytree=0.8,min_child_weight=5,reg_lambda=10,random_state=42,n_jobs=2,tree_method='hist')
res={k:[] for k in ['raw','raw+hung','centered','centered+hung']}
for f,(a,b) in enumerate(GroupKFold(5).split(tr,y,tr.prompt_id)):
    A,B=tr.iloc[a].reset_index(drop=True),tr.iloc[b].reset_index(drop=True)
    tf=TfidfVectorizer(analyzer='char',ngram_range=(3,5),lowercase=False,min_df=2,max_features=30000,sublinear_tf=True); Ta=tf.fit_transform(A.text); Tb=tf.transform(B.text)
    svd=TruncatedSVD(100,random_state=42).fit(Ta); cols=[f's{i}' for i in range(100)]
    Xa=pd.concat([B32(A.text),fw(A.text),pd.DataFrame(svd.transform(Ta),columns=cols)],axis=1)
    Xb=pd.concat([B32(B.text),fw(B.text),pd.DataFrame(svd.transform(Tb),columns=cols)],axis=1)
    for tag,(xa,xb) in {'raw':(Xa,Xb),'centered':(center(Xa,A.prompt_id.values),center(Xb,B.prompt_id.values))}.items():
        m=xgb.XGBClassifier(**P0).fit(xa,y[a]); P=m.predict_proba(xb)
        res[tag].append(balanced_accuracy_score(y[b],P.argmax(1))); res[tag+'+hung'].append(balanced_accuracy_score(y[b],hung(P,B.prompt_id.values)))
    print('fold',f,{k:round(v[-1],3) for k,v in res.items()},flush=True)
print({k:f'{np.mean(v):.3f} ± {np.std(v):.3f}' for k,v in res.items()})
