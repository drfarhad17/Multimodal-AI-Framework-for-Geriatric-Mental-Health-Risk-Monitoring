"""Optional genuine SHAP/LIME explanations of saved RF or probability-fusion models."""
import argparse,json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from generate_synthetic_data import FEATURES,CLASSES

def main(a):
    import shap
    from lime.lime_tabular import LimeTabularExplainer
    out=Path(a.results); d=np.load(out/'explanation_inputs.npz'); train=d['train']; test=d['test']; shape=train.shape[1:]
    rf=joblib.load(out/'RandomForest.joblib'); weights_path=out/'fusion_weights.json'
    models={}; weights=json.loads(weights_path.read_text()) if weights_path.exists() else {'RandomForest':1.0}
    if len(weights)>1:
        import torch
        from models import Network
        for name in weights:
            if name=='RandomForest': continue
            net=Network(name,shape[-1]); net.load_state_dict(torch.load(out/f'{name}.pt',map_location='cpu',weights_only=True)); net.eval(); models[name]=net
    def predict(flat):
        result=weights['RandomForest']*rf.predict_proba(flat)
        if models:
            import torch
            with torch.no_grad():
                for name,net in models.items(): result+=weights[name]*torch.softmax(net(torch.tensor(flat.reshape(-1,*shape),dtype=torch.float32)),1).numpy()
        return result
    flat=train.reshape(len(train),-1); cases=test[:a.samples].reshape(min(a.samples,len(test)),-1)
    names=[f'session_{t}_{f}' for t in range(10) for f in FEATURES]
    rng=np.random.default_rng(42); bg=flat[rng.choice(len(flat),min(20,len(flat)),replace=False)]
    np.random.seed(42)
    explainer=shap.KernelExplainer(predict,bg); values=explainer.shap_values(cases,nsamples=a.shap_samples)
    if isinstance(values,list): values=np.stack(values,axis=-1)
    values=np.asarray(values); np.save(out/'shap_values.npy',values)
    # Average absolute attribution over explained participants and classes, then sum sessions.
    importance=np.abs(values).mean(axis=(0,2)).reshape(10,len(FEATURES)).sum(0)
    pd.DataFrame({'feature':FEATURES,'mean_absolute_shap_sum_sessions':importance}).sort_values('mean_absolute_shap_sum_sessions',ascending=False).to_csv(out/'shap_importance.csv',index=False)
    lime=LimeTabularExplainer(flat,feature_names=names,class_names=CLASSES,mode='classification',random_state=42)
    explanation=lime.explain_instance(cases[0],predict,num_features=15,top_labels=1,num_samples=a.lime_samples)
    explanation.save_to_file(str(out/'lime_example.html'))
    (out/'explanation_metadata.json').write_text(json.dumps({'explained_participants':len(cases),'model':'fusion' if len(weights)>1 else 'RF','shap_background_n':len(bg),'shap_nsamples':a.shap_samples,'caution':'Small sample approximate post-hoc explanations; not clinician validated. Perturbations can break temporal correlations.'},indent=2))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--results',default='results');p.add_argument('--samples',type=int,default=5);p.add_argument('--shap-samples',type=int,default=512);p.add_argument('--lime-samples',type=int,default=1000);main(p.parse_args())
