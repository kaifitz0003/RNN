"""
In the code below, we aim to compare torch.nn.RNN with our manual RNN, and with torch.nn.RNNCell.

The data starts as 3d (X_NLH), in the shape NLH.
N is each day such as Monday, Tuesday, etc.
L are the measurement times, such as the Morning, evening, and nightime.
H are the features, temerature and humidity.
We then change the 3d dataset into 2d (X_LH), by picking only 1 day.

In the RNN, the hidden state is the RNN's memory.
In our case, we picked Hout=5, meaning each measurment time (L) is represented by 5 numbers.

Our code include 3 models, torch.nn.RNN, a manual RNN, and torch.nn.RNN_CELL.
All 3 of these models share the same weights, so the models produce the same output.

In the RNN, we simply plugged our input into torch.nn.RNN, and got our output.

In our manual RNN, we need to loop over all the measurments times.
Additionally, we hold on to the hidden state, and update it at every loop iteration.

We do the same thing in the RNN_Cell, loop over the measurment times, and update the hidden state.
"""
import torch
import torch.nn as nn
#torch.manual_seed(0)

### DATA ################################
#                        T, H    (Temperature, Humidity)
X_NLH = torch.tensor([
                      # Monday
                      [[70,40], # Morning
                       [75,43], # Evening
                       [72,39]], # Nightime
                      # Tuesday
                      [[30,10],
                       [32,11],
                       [35,9]],
                      # Wednesday
                      [[77,45],
                       [69,41],
                       [67,40]],
                      # Thursday
                      [[30,13],
                       [34,11],
                       [36,10]]], dtype=torch.float32)/1000

# N, L, Hin = X.shape
X_LH = X_NLH[0] # Pick the first day, Monday.
L, Hin = X_LH.shape


### MODEL WEIGHTS ###########################
Hout = 5 # Number of hidden states.
h0 = torch.zeros(1,Hout)
WX = torch.tensor([[0.1, 0.2],
                  [0.3, 0.4],
                  [0.5, 0.6],
                  [0.7, 0.8],
                  [0.9, 1.0]]) # (hidden_size=Hout, input_size=Hin) = (5,2). We picked Hout to be 5. Hin is 2, which comes from the input.
WH = torch.tensor([[0.1, 0.2, 0.3, 0.4, 0.5],
                  [0.5, 0.4, 0.3, 0.2, 0.1],
                  [0.2, 0.3, 0.4, 0.5, 0.6],
                  [0.6, 0.5, 0.4, 0.3, 0.2],
                  [0.1, 0.3, 0.5, 0.7, 0.9]]) # (hidden_size=Hout, hidden_size=Hout)
BX = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5]) # (hidden_size=Hout)
BH = torch.tensor([0.5, 0.4, 0.3, 0.2, 0.1]) # (hidden_size=Hout)


### Model #1: RNN ##############################
rnn = nn.RNN(input_size=Hin, hidden_size=Hout) # Batch_First is only used when there is an N(3d).
# Update RNN weights
rnn.weight_ih_l0.data = WX
rnn.weight_hh_l0.data = WH
rnn.bias_ih_l0.data   = BX
rnn.bias_hh_l0.data   = BH
# Pass X through RNN
output, ht_rnn = rnn(X_LH)
#print(ht_rnn)
#print(output)

### Model #2: RNN FROM SCRATCH ###################
ht = h0
for t in range(L):
  WX_dot = X_LH[t]@WX.T # 3,5 (L, Hout))
  #print(WX_dot)
  WH_dot = ht@WH.T # 1x5
  #print(WH_dot)
  final = WX_dot+WH_dot+BX+BH
  ht = torch.tanh(final)
  #print(ht)

  print(torch.allclose(output[t],ht)) # output contains 3 rows, for morning, evening, and nighttime measurements. ht contains the most recent state.

### MODEL #3: RNN_CELL ############################
hx = h0.flatten() # For the rnn_cell, h needs to be 1d.
rnn_cell = torch.nn.RNNCell(input_size=Hin, hidden_size=Hout)

rnn_cell.weight_ih.data = WX
rnn_cell.weight_hh.data = WH
rnn_cell.bias_ih.data   = BX
rnn_cell.bias_hh.data   = BH
for t in range(L):
  hx = rnn_cell(X_LH[t],hx)
  #print(hx)
  print(torch.allclose(output[t], hx))
