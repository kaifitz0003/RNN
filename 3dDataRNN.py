"""
In the code below, we aim to compare torch.nn.RNN with our manual RNN, and with torch.nn.RNNCell.

The data starts as 3D (X_NLH), in the shape NLH.
N is each day such as Monday, Tuesday, etc.
L are the measurement times, such as the Morning, Evening, and Nighttime.
H are the features, temperature and humidity.

In the RNN, the hidden state is the RNN's memory.
In our case, we picked Hout=5, meaning each measurement time (L) is represented by 5 numbers.

Our code includes 3 models, torch.nn.RNN, a manual RNN, and torch.nn.RNNCell.
All 3 of these models share the same weights, so the models produce the same output.

In the RNN, we simply plugged our input into torch.nn.RNN, and got our output.

In our manual RNN, we need to loop over all the measurement times.
Additionally, we hold on to the hidden state, and update it at every loop iteration.

We do the same thing in the RNNCell, loop over the measurement times, and update the hidden state.

This version works with 3D data (multiple days), so it generalizes the previous 2D (single day) RNN.
"""
import torch
import torch.nn as nn

### DATA ################################
X_NLH = torch.tensor([
    # Monday
    [[70,40],[75,43],[72,39]],
    # Tuesday
    [[30,10],[32,11],[35,9]],
    # Wednesday
    [[77,45],[69,41],[67,40]],
    # Thursday
    [[30,13],[34,11],[36,10]]
], dtype=torch.float32)/1000

N, L, Hin = X_NLH.shape

### MODEL WEIGHTS ###########################
Hout = 5
h0 = torch.zeros(N,Hout)
WX = torch.tensor([[0.1, 0.2],
                   [0.3, 0.4],
                   [0.5, 0.6],
                   [0.7, 0.8],
                   [0.9, 1.0]])
WH = torch.tensor([[0.1, 0.2, 0.3, 0.4, 0.5],
                   [0.5, 0.4, 0.3, 0.2, 0.1],
                   [0.2, 0.3, 0.4, 0.5, 0.6],
                   [0.6, 0.5, 0.4, 0.3, 0.2],
                   [0.1, 0.3, 0.5, 0.7, 0.9]])
BX = torch.tensor([0.1,0.2,0.3,0.4,0.5])
BH = torch.tensor([0.5,0.4,0.3,0.2,0.1])

### Model #1: RNN ##############################
rnn = nn.RNN(input_size=Hin, hidden_size=Hout, batch_first=True)
rnn.weight_ih_l0.data = WX
rnn.weight_hh_l0.data = WH
rnn.bias_ih_l0.data   = BX
rnn.bias_hh_l0.data   = BH

output, ht_rnn = rnn(X_NLH)

### Model #2: Manual RNN ###################
ht = h0
outputs_manual = []

for t in range(L):
    WX_dot = X_NLH[:, t, :] @ WX.T
    WH_dot = ht @ WH.T
    final = WX_dot + WH_dot + BX + BH
    ht = torch.tanh(final)
    outputs_manual.append(ht.unsqueeze(1))

outputs_manual = torch.cat(outputs_manual, dim=1)
print(torch.allclose(output, outputs_manual))

### Model #3: RNNCell #######################
hx = h0
rnn_cell = nn.RNNCell(input_size=Hin, hidden_size=Hout)
rnn_cell.weight_ih.data = WX
rnn_cell.weight_hh.data = WH
rnn_cell.bias_ih.data   = BX
rnn_cell.bias_hh.data   = BH

outputs_cell = []
for t in range(L):
    WX_dot = X_NLH[:, t, :] @ WX.T
    WH_dot = hx @ WH.T
    final = WX_dot + WH_dot + BX + BH
    hx = torch.tanh(final)
    outputs_cell.append(hx.unsqueeze(1))

outputs_cell = torch.cat(outputs_cell, dim=1)
print(torch.allclose(output, outputs_cell))
h0
