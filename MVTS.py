"""
In the code below, we use Multi-variate time series classification to classify the temperature and humidity on a given day.

We used torch.rand() to add noise to the dataset, for both good and bad days.
Our data is in N,L,Hin format, with the 2 classes(num_classes), for good and bad days.
The data consists of 10(L) daily readings for temperature and humidity(Hin), over the span of 100(N) days.
Hout is the hidden size of our model. It determines how many features the hidden state has. Each day will have this many features.

"""

import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader,TensorDataset
from tqdm import tqdm
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

Hout=5
num_epochs=50
lr=0.01

### DATA
N,L,Hin = 100,10,2 # Temperature, Humidity, Wind speed, and rain chance. Add this to data.
batch_size=4
num_classes = 2

X = torch.zeros(N,L,Hin)
# Not L, N, Hin, but N, L, Hin.
X[:N//2,:,0] = 60 + X[:N//2,:,0] + torch.rand(N//2,L)*9
X[N//2:,:,0] = 30 + X[N//2:,:,0] + torch.rand(N//2,L)*9
X[:N//2,:,1] = 10 + X[:N//2,:,1] + torch.rand(N//2,L)*9
X[N//2:,:,1] = 80 + X[N//2:,:,1] + torch.rand(N//2,L)*9
X = X/100 # The datapoints are made smaller(simple normalization).
y=torch.cat((torch.zeros(N//2),torch.ones(N//2))).long()
X_train,X_test,y_train,y_test = train_test_split(X,y)

N_train,L,Hin=X_train.shape
train_dataset = TensorDataset(X_train,y_train)
train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)
test_dataset = TensorDataset(X_test,y_test)
test_dataloader = DataLoader(test_dataset, shuffle=True, batch_size=batch_size)

### MODEL
backbone = nn.RNN(input_size=Hin,hidden_size=Hout, batch_first=True)
head = nn.Linear(in_features=Hout, out_features=num_classes)
model = nn.ModuleList([backbone,head]).to(device)
### TRAINING
model.train()
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr = lr)
for epoch in tqdm(range(num_epochs)):
  for Xb, yb in train_dataloader:
    Xb, yb = Xb.to(device), yb.to(device)
    out,h = backbone(Xb)
    logits = head(h.squeeze(0))
    loss = loss_fn(logits, yb)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
#print(out,h)

### TESTING
score,counter=0,0
model.eval()
with torch.no_grad():
  for Xb, yb in tqdm(test_dataloader):
    Xb, yb = Xb.to(device), yb.to(device)
    out,h = backbone(Xb)
    logits = head(h.squeeze(0))
    y_pred = torch.argmax(logits, dim=1)
    print(yb, y_pred)
    score+=(yb==y_pred).sum().item()
    counter+=len(yb)

  print((score/counter)*100)
