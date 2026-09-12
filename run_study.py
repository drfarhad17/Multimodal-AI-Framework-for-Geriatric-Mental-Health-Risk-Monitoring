"""Participant-independent benchmark. All reported outputs are computed from predictions."""
import argparse, hashlib, json, platform, time
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,roc_auc_score,average_precision_score,matthews_corrcoef,confusion_matrix,log_loss
from generate_synthetic_data import generate, FEATURES, GROUPS, CLASSES

def metrics(y,p):
    pred=p.argmax(1); hot=np.eye(5)[y]
    return dict(accuracy=accuracy_score(y,pred),precision_macro=precision_score(y,pred,average='macro',zero_division=0),
      recall_macro=recall_score(y,pred,average='macro',zero_division=0),f1_macro=f1_score(y,pred,average='macro',zero_division=0),
      auc_macro_ovr=roc_auc_score(hot,p,average='macro') if len(np.unique(y))==5 else None,
      average_precision_macro=average_precision_score(hot,p,average='macro'),mcc=matthews_corrcoef(y,pred),
      brier_multiclass_sum=float(np.mean(np.sum((p-hot)**2,axis=1))),
      probability_mae=float(np.mean(abs(p-hot))),probability_rmse=float(np.sqrt(np.mean((p-hot)**2))))

def transform_fit(x,train):
    imp=SimpleImputer(strategy='median'); sc=MinMaxScaler()
    z=imp.fit_transform(x[train].reshape(-1,x.shape[-1])); sc.fit(z)
    return sc.transform(imp.transform(x.reshape(-1,x.shape[-1]))).reshape(x.shape).astype('float32'),imp,sc

def learn_weights(y,ps):
    result=minimize(lambda w:log_loss(y,np.clip(np.einsum('m,mnk->nk',w,ps),1e-12,1),labels=np.arange(5)),np.ones(len(ps))/len(ps),
      bounds=[(0,1)]*len(ps),constraints={'type':'eq','fun':lambda w:w.sum()-1},method='SLSQP')
    if not result.success: raise RuntimeError(result.message)
    return result.x/result.x.sum()

def main(a):
    out=Path(a.out); out.mkdir(parents=True,exist_ok=True); data=Path(a.data)
    if not (data/'participants.csv').exists(): generate(data,a.participants,a.seed)
    p=pd.read_csv(data/'participants.csv').sort_values('participant_id').reset_index(drop=True)
    df=pd.read_csv(data/'synthetic_raw_data.csv').sort_values(['participant_id','session'])
    assert list(df.participant_id.unique())==list(p.participant_id)
    assert df.groupby('participant_id').size().eq(10).all()
    assert not df.duplicated(['participant_id','session']).any()
    raw=df[FEATURES].to_numpy().reshape(len(p),10,-1); missing=np.isnan(raw).mean((1,2))
    # Offline interpolation within each participant; limit to short interior gaps.
    x=np.stack([pd.DataFrame(v).interpolate(limit=2,limit_area='inside').to_numpy() for v in raw])
    y=p.label.to_numpy(); ix=np.arange(len(y))
    tr,hold=train_test_split(ix,test_size=.3,stratify=y,random_state=a.seed)
    va,te=train_test_split(hold,test_size=.5,stratify=y[hold],random_state=a.seed)
    assert not(set(tr)&set(te) or set(tr)&set(va) or set(te)&set(va))
    splits=p[['participant_id','label']].copy(); splits['split']='train'; splits.loc[va,'split']='validation'; splits.loc[te,'split']='test'; splits.to_csv(out/'splits.csv',index=False)
    z,imp,sc=transform_fit(x,tr); joblib.dump({'imputer':imp,'scaler':sc,'features':FEATURES},out/'preprocessing.joblib')
    names=['RandomForest'] if a.mode=='rf' else ['RandomForest','LSTM','CNN','Transformer']
    vp=[]; tp=[]; models={}; results=[]; timings=[]
    for name in names:
        start=time.perf_counter()
        if name=='RandomForest':
            model=RandomForestClassifier(n_estimators=200,min_samples_leaf=2,random_state=a.seed,n_jobs=2)
            model.fit(z[tr].reshape(len(tr),-1),y[tr]); predict=lambda q:model.predict_proba(q.reshape(len(q),-1))
            joblib.dump(model,out/'RandomForest.joblib')
        else:
            from models import TorchClassifier
            model=TorchClassifier(name,a.epochs,a.seed).fit(z[tr],y[tr]); predict=model.predict_proba
            import torch
            torch.save(model.net.state_dict(),out/f'{name}.pt')
        elapsed=time.perf_counter()-start; v=predict(z[va]); start=time.perf_counter(); t=predict(z[te]); infer=time.perf_counter()-start
        vp.append(v);tp.append(t);models[name]=model
        results.append({'model':name,**metrics(y[te],t)})
        timings.append({'model':name,'training_seconds':elapsed,'test_batch_seconds':infer,'test_participants':len(te)})
    if len(names)>1:
        w=learn_weights(y[va],np.array(vp)); final=np.einsum('m,mnk->nk',w,tp)
        results.append({'model':'ProbabilityFusion',**metrics(y[te],final)})
        (out/'fusion_weights.json').write_text(json.dumps(dict(zip(names,w)),indent=2))
        for drop in range(len(names)):
            keep=[i for i in range(len(names)) if i!=drop]; aw=learn_weights(y[va],np.array(vp)[keep]); ap=np.einsum('m,mnk->nk',aw,np.array(tp)[keep])
            results.append({'model':'Fusion_without_'+names[drop],**metrics(y[te],ap)})
        results.append({'model':'Fusion_without_posthoc_XAI',**metrics(y[te],final)})
    else: final=tp[0]
    # Retrained RF modality ablations, explicitly distinct from fusion ablations.
    for group,cols in GROUPS.items():
        keep=[i for i,f in enumerate(FEATURES) if f not in cols]; q=z[:,:,keep].reshape(len(z),-1)
        model=RandomForestClassifier(n_estimators=100,min_samples_leaf=2,random_state=a.seed,n_jobs=2).fit(q[tr],y[tr])
        results.append({'model':'RF_without_'+group,**metrics(y[te],model.predict_proba(q[te]))})
    pd.DataFrame(results).to_csv(out/'metrics.csv',index=False); pd.DataFrame(timings).to_csv(out/'timings.csv',index=False)
    prediction=p.iloc[te][['participant_id','age','sex','label']].copy(); prediction['predicted_label']=final.argmax(1)
    for k,c in enumerate(CLASSES): prediction['p_'+c]=final[:,k]
    prediction.to_csv(out/'test_predictions.csv',index=False)
    cm=confusion_matrix(y[te],final.argmax(1),labels=range(5)); pd.DataFrame(cm,index=CLASSES,columns=CLASSES).to_csv(out/'confusion_matrix.csv')
    perclass=[]
    for k,c in enumerate(CLASSES):
        tp0=cm[k,k]; fn=cm[k].sum()-tp0; fp=cm[:,k].sum()-tp0; tn=cm.sum()-tp0-fn-fp
        perclass.append(dict(class_name=c,sensitivity=tp0/(tp0+fn),specificity=tn/(tn+fp)))
    pd.DataFrame(perclass).to_csv(out/'class_metrics.csv',index=False)
    rng=np.random.default_rng(a.seed); boot=[]
    for _ in range(a.bootstrap):
        b=rng.integers(0,len(te),len(te)); boot.append(metrics(y[te][b],final[b]))
    bdf=pd.DataFrame(boot); pd.DataFrame({'metric':bdf.columns,'lower_95':bdf.quantile(.025).values,'upper_95':bdf.quantile(.975).values}).to_csv(out/'bootstrap_ci.csv',index=False)
    subgroup=[]
    for label,mask in [('age60_69',p.iloc[te].age.to_numpy()<70),('age70_79',(p.iloc[te].age.to_numpy()>=70)&(p.iloc[te].age.to_numpy()<80)),('age80_90',p.iloc[te].age.to_numpy()>=80),('female',p.iloc[te].sex.to_numpy()=='female'),('male',p.iloc[te].sex.to_numpy()=='male'),('missing_le_5pct',missing[te]<=.05),('missing_gt_5pct',missing[te]>.05)]:
        if mask.sum(): subgroup.append({'group':label,'n':int(mask.sum()),**metrics(y[te][mask],final[mask])})
    pd.DataFrame(subgroup).to_csv(out/'subgroups.csv',index=False)
    # Explicit binary event: any non-normal synthetic reference label.
    event=(y[te]!=0).astype(int); risk=1-final[:,0]; calibration=[]; dca=[]
    for lo in np.arange(0,1,.1):
        mask=(risk>=lo)&(risk<(lo+.1)) if lo<.9 else (risk>=lo)&(risk<=1)
        if mask.any(): calibration.append({'bin_start':lo,'n':int(mask.sum()),'mean_probability':risk[mask].mean(),'observed_frequency':event[mask].mean()})
    for t in np.arange(.05,.85,.05):
        pos=risk>=t; nb=((pos&(event==1)).sum()-(pos&(event==0)).sum()*t/(1-t))/len(te)
        dca.append({'threshold':t,'model_net_benefit':nb,'treat_all':event.mean()-(1-event.mean())*t/(1-t),'treat_none':0})
    pd.DataFrame(calibration).to_csv(out/'calibration_any_condition.csv',index=False); pd.DataFrame(dca).to_csv(out/'decision_curve_any_condition.csv',index=False)
    if a.cv:
        cv=[]
        for fold,(fit,val) in enumerate(StratifiedKFold(5,shuffle=True,random_state=a.seed).split(tr,y[tr]),1):
            fi,vi=tr[fit],tr[val]; zz,_,_=transform_fit(x,fi); zz=zz.reshape(len(x),-1)
            model=RandomForestClassifier(n_estimators=100,min_samples_leaf=2,random_state=a.seed,n_jobs=2).fit(zz[fi],y[fi]); cv.append({'fold':fold,**metrics(y[vi],model.predict_proba(zz[vi]))})
        pd.DataFrame(cv).to_csv(out/'rf_training_cv.csv',index=False)
    np.savez_compressed(out/'explanation_inputs.npz',train=z[tr],test=z[te],test_ids=p.iloc[te].participant_id.to_numpy(dtype=str))
    report={'status':'New reference implementation; NOT reproduction of manuscript numerical results','arguments':vars(a),'python':platform.python_version(),'platform':platform.platform(),'processor':platform.processor(),'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'test_n':len(te),'binary_event':'any non-normal label','binary_brier':float(np.mean((risk-event)**2))}
    (out/'run_metadata.json').write_text(json.dumps(report,indent=2)); print(pd.DataFrame(results)[['model','accuracy','f1_macro']].to_string(index=False))

if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('--data',default='data'); ap.add_argument('--out',default='results'); ap.add_argument('--mode',choices=['rf','full'],default='rf'); ap.add_argument('--participants',type=int,default=2000); ap.add_argument('--seed',type=int,default=42); ap.add_argument('--epochs',type=int,default=30); ap.add_argument('--bootstrap',type=int,default=1000); ap.add_argument('--cv',action='store_true'); main(ap.parse_args())
