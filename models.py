"""Optional PyTorch feature-sequence baselines, not BERT/raw-audio implementations."""
import numpy as np
import torch
from torch import nn

class Network(nn.Module):
    def __init__(self,kind,n_features):
        super().__init__(); self.kind=kind
        if kind=='LSTM': self.encoder=nn.LSTM(n_features,128,batch_first=True); dim=128
        elif kind=='CNN':
            self.encoder=nn.Sequential(nn.Conv2d(1,16,3,padding=1),nn.ReLU(),nn.MaxPool2d(2),nn.AdaptiveAvgPool2d((1,1))); dim=16
        else:
            self.project=nn.Linear(n_features,32)
            self.position=nn.Parameter(torch.zeros(1,10,32))
            self.encoder=nn.TransformerEncoder(nn.TransformerEncoderLayer(32,4,64,dropout=.1,batch_first=True),2); dim=32
        self.head=nn.Linear(dim,5)
    def forward(self,x):
        if self.kind=='LSTM': z=self.encoder(x)[0][:,-1]
        elif self.kind=='CNN': z=self.encoder(x.unsqueeze(1)).flatten(1)
        else: z=self.encoder(self.project(x)+self.position[:,:x.shape[1]]).mean(1)
        return self.head(z)

class TorchClassifier:
    def __init__(self,kind,epochs=30,seed=42): self.kind,self.epochs,self.seed=kind,epochs,seed
    def fit(self,x,y):
        torch.manual_seed(self.seed); torch.set_num_threads(2)
        torch.use_deterministic_algorithms(True)
        self.net=Network(self.kind,x.shape[-1]); optimizer=torch.optim.Adam(self.net.parameters(),lr=.001)
        ds=torch.utils.data.TensorDataset(torch.tensor(x,dtype=torch.float32),torch.tensor(y,dtype=torch.long))
        loader=torch.utils.data.DataLoader(ds,batch_size=32,shuffle=True,generator=torch.Generator().manual_seed(self.seed))
        self.net.train()
        for _ in range(self.epochs):
            for bx,by in loader:
                optimizer.zero_grad(); loss=nn.functional.cross_entropy(self.net(bx),by); loss.backward(); optimizer.step()
        return self
    def predict_proba(self,x):
        self.net.eval()
        with torch.no_grad(): return torch.softmax(self.net(torch.tensor(x,dtype=torch.float32)),1).numpy()
