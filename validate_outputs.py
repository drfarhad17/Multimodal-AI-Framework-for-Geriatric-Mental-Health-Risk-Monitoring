"""Independent integrity checks for the default reference run."""
import tempfile,hashlib
from pathlib import Path
import numpy as np
import pandas as pd
from generate_synthetic_data import generate,label_scores
root=Path(__file__).parent
p=pd.read_csv(root/'data/participants.csv'); d=pd.read_csv(root/'data/synthetic_raw_data.csv')
assert len(p)==2000 and len(d)==20000
assert p.label.value_counts().eq(400).all()
assert [label_scores(r.gds15,r.gad7,r.mmse,r.moca) for r in p.itertuples()]==p.label.tolist()
s=pd.read_csv(root/'results_verified/splits.csv')
assert s.participant_id.is_unique and s.split.value_counts().to_dict()=={'train':1400,'test':300,'validation':300}
q=pd.read_csv(root/'results_verified/test_predictions.csv')
assert set(q.participant_id)==set(s.loc[s.split=='test','participant_id'])
probs=q.filter(regex='^p_').to_numpy(); assert np.allclose(probs.sum(1),1)
cm=pd.read_csv(root/'results_verified/confusion_matrix.csv',index_col=0).to_numpy()
assert cm.sum()==300
m=pd.read_csv(root/'results_verified/metrics.csv').iloc[0]
assert np.isclose(np.trace(cm)/cm.sum(),m.accuracy)
with tempfile.TemporaryDirectory() as tmp:
    generate(tmp,2000,42)
    for name in ['participants.csv','synthetic_raw_data.csv']:
        assert hashlib.sha256((Path(tmp)/name).read_bytes()).digest()==hashlib.sha256((root/'data'/name).read_bytes()).digest()
print('PASS: dataset counts, label rules, split identities, probability normalization, metric consistency, deterministic CSV regeneration')
