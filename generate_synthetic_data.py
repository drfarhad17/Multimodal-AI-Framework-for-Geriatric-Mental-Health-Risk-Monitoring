"""New, documented conditional simulator; not the manuscript's original dataset."""
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd

CLASSES = ['normal', 'depression', 'anxiety', 'cognitive_decline', 'dementia']
GROUPS = {'physiological':['hrv_ms','systolic_bp','oxygen_pct','stress'],
          'behavioral':['sleep_hours','mobility','social_interaction','activity_regularity'],
          'speech':['emotional_tone','pause_seconds','speech_rate'],
          'conversational':['semantic_coherence','conversation_consistency']}
FEATURES = sum(GROUPS.values(), [])
RATES = dict(zip(GROUPS, [.048,.062,.071,.056]))

def label_scores(gds,gad,mmse,moca):
    # Highest normalized severity among eligible categories. Normalization is an assumption.
    s=[0, (gds/15 if gds>=6 else -1), (gad/21 if gad>=10 else -1),
       (max((30-mmse)/30,(30-moca)/30) if 21<=mmse<=24 or 18<=moca<=25 else -1),
       ((30-mmse)/30 if mmse<=20 else -1)]
    return int(np.argmax(s))

def generate(out, n=2000, seed=42):
    if n%5 or n<100: raise ValueError('participants must be >=100 and divisible by 5')
    out=Path(out); out.mkdir(parents=True,exist_ok=True)
    rng=np.random.default_rng(seed)
    # Rejection sampling yields exact class balance while retaining overlapping score rules.
    scores=[]
    for k in range(5):
        accepted=0
        while accepted<n//5:
            g,a,m,c=[int(rng.integers(lo,hi)) for lo,hi in [(0,16),(0,22),(10,31),(10,31)]]
            if label_scores(g,a,m,c)==k:
                scores.append((g,a,m,c,k)); accepted+=1
    rng.shuffle(scores)
    sex=np.array(['female']*round(n*.52)+['male']*(n-round(n*.52))); rng.shuffle(sex)
    ages=rng.normal(74.6,8.1,n)
    while np.any((ages<60)|(ages>90)):
        bad=(ages<60)|(ages>90); ages[bad]=rng.normal(74.6,8.1,bad.sum())
    participants=[]; rows=[]
    for i,(g,a,m,c,k) in enumerate(scores):
        pid=f'SYN_{i:05d}'
        participants.append([pid,round(ages[i],2),sex[i],g,a,m,c,k,CLASSES[k]])
        d,an,cog=g/15,a/21,(60-m-c)/40
        base=np.array([55-14*an,125+12*an,97-1.2*cog, .15+.5*an,
            7.5-1.4*d-.5*an,.8-.35*cog-.2*d,.8-.5*d,.85-.35*cog,
            .75-.45*d, .4+1.1*cog,145-25*cog-15*d,.9-.5*cog,.85-.4*cog])
        scale=np.array([8,10,.7,.12,.65,.13,.14,.12,.13,.2,12,.12,.12])
        random_effect=rng.normal(size=len(FEATURES))*.6*scale
        state=np.zeros(len(FEATURES))
        for t in range(10):
            state=.65*state+rng.normal(size=len(FEATURES))*.76*scale
            x=base+random_effect+state
            x=np.clip(x,[5,80,85,0,2,0,0,0,0,0,40,0,0],[120,200,100,1,12,1,1,1,1,5,220,1,1])
            rows.append([pid,t,f'2026-01-01T00:{5*t:02d}:00',*x])
    p=pd.DataFrame(participants,columns=['participant_id','age','sex','gds15','gad7','mmse','moca','label','class_name'])
    df=pd.DataFrame(rows,columns=['participant_id','session','timestamp',*FEATURES])
    for group,cols in GROUPS.items():
        # At most two missing sessions per modality keeps all participants within the 20% exclusion rule.
        for i in range(n):
            count=min(int(rng.binomial(10,RATES[group])),2)
            ix=i*10+rng.choice(10,count,replace=False)
            df.loc[ix,cols]=np.nan
    p.to_csv(out/'participants.csv',index=False); df.to_csv(out/'synthetic_raw_data.csv',index=False)
    meta={'status':'NEW reference simulation, not original study data','seed':seed,'participants':n,'sessions':len(df),
      'actual_age_mean':float(p.age.mean()),'actual_age_sd':float(p.age.std()),'missing_rates':{g:float(df[c].isna().mean().mean()) for g,c in GROUPS.items()},
      'severity_normalization':'gds/15, gad/21, max(30-mmse,30-moca)/30; see label_scores',
      'missingness':'modality-block missingness, binomial counts capped at 2 per participant; actual rates differ from target'}
    (out/'generation_metadata.json').write_text(json.dumps(meta,indent=2))
    units=['ms','mmHg','percent','unit interval','hours','unit interval','unit interval','unit interval','unit interval','seconds','words/min','unit interval','unit interval']
    pd.DataFrame({'feature':FEATURES,'unit':units,'description':['Simulated '+x.replace('_',' ') for x in FEATURES]}).to_csv(out/'data_dictionary.csv',index=False)
    return p,df

if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('--out',default='data'); ap.add_argument('--participants',type=int,default=2000); ap.add_argument('--seed',type=int,default=42)
    a=ap.parse_args(); generate(a.out,a.participants,a.seed)
