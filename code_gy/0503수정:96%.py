#test_set에 경로지정부터 train으로 잘못되어있었음 그니까 training시키고 그 데이터로 test한 것.
#test_set으로 test했을때

import torch
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch.nn.init 


train_data = pd.read_csv('./train.csv', usecols=['rainfall_all',
                        'avgtemp_all','humidity_all','GDP','max_power'])
trainxy_data = train_data.values
train_set = trainxy_data [::-1] 

test_data = pd.read_csv('./test.csv', usecols=['rainfall_all', 
                        'avgtemp_all','humidity_all','GDP','max_power'])
testxy_data = test_data.values
test_set = testxy_data [::-1]  # train으로 되어있었음. test자체가 안된것.
print(test_set.shape,"Y")

device = 'cuda' if torch.cuda.is_available() else 'cpu'

torch.manual_seed(0)


seq_length = 7 
data_parameter = 5
hidden_dim = 10 
output_dim = 1 
learning_rate = 0.01
iterations = 500 #500에서 오래걸려서 일시적으로 변경한 것


class Power(torch.nn.Module): 
    def __init__(self, input_dim, hidden_dim, output_dim, layers):
        super(Power, self).__init__()
        self.layer1 = torch.nn.Sequential(
            torch.nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1), 
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = torch.nn.Sequential(
            torch.nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2))
        self.fc_cnn = torch.nn.Linear(64, 7, bias=True)
        torch.nn.init.xavier_uniform_(self.fc_cnn.weight)

        self.LSTM = torch.nn.LSTM(input_dim, hidden_dim, num_layers=layers, batch_first=True) 
        self.fc_rnn = torch.nn.Linear(hidden_dim, output_dim, bias=True) 

    def forward(self, x):
        x = torch.unsqueeze(x,1)
        w,h = x.shape[2],x.shape[3] 
        out = self.layer1(x) 
        out = self.layer2(out)
        out = out.view(out.size(0), -1) 
        out = self.fc_cnn(out)
        out = torch.unsqueeze(out,-1)
        x, _status = self.LSTM(out) 
        x = self.fc_rnn(x[:, -1])
        return x

power_prediction = Power(1, hidden_dim, output_dim, 1)

criterion = torch.nn.MSELoss() 

optimizer = optim.Adam(power_prediction.parameters(), lr=learning_rate) 


def build_dataset(time_series, seq_length):
  dataX = []
  dataY = []
  for i in range(0, len(time_series)-seq_length): 
    _x = time_series[i:i + seq_length, :-1] 
    _y = time_series[i+seq_length, [-1]] 
    
    dataX.append(_x)
    dataY.append(_y)
    
  return np.array(dataX), np.array(dataY) 

trainX, trainY = build_dataset(train_set, seq_length)
testX, testY = build_dataset(test_set, seq_length) 

train_x_numerator = trainX - np.min(trainX, 0) 
train_x_denominator = np.max(trainX, 0) - np.min(trainX, 0)
train_x_set = train_x_numerator / (train_x_denominator + 1e-7)

train_y_numerator = trainY - np.min(trainY, 0) 
train_y_denominator = np.max(trainY, 0) - np.min(trainY, 0)
train_y_set = train_y_numerator / (train_y_denominator + 1e-7)

test_x_numerator = testX - np.min(testX, 0) 
test_x_denominator = np.max(testX, 0) - np.min(testX, 0)
test_x_set = test_x_numerator / (test_x_denominator + 1e-7)

test_y_numerator = testY - np.min(testY, 0) 
test_y_denominator = np.max(testY, 0) - np.min(testY, 0)
test_y_set = test_y_numerator / (test_y_denominator + 1e-7)


trainX_tensor = torch.FloatTensor(train_x_set) 
trainY_tensor = torch.FloatTensor(train_y_set)

testX_tensor = torch.FloatTensor(test_x_set)
testY_tensor = torch.FloatTensor(test_y_set)

criterion = torch.nn.MSELoss() 

optimizer = optim.Adam(power_prediction.parameters(), lr=learning_rate) 

for i in range(iterations+1): 
    X = trainX_tensor
    Y = trainY_tensor
    
    optimizer.zero_grad()
    outputs = power_prediction(X)

    loss = criterion(outputs, Y) 
    loss.backward() 
    optimizer.step()
    if i % 100 == 0 :
      print(i, "loss: ", loss.item()) 


with torch.no_grad():
  
    prediction = power_prediction(testX_tensor)
    error = torch.nn.MSELoss(reduction='sum')
    sum_mse = error(prediction, testY_tensor)
    print(testY_tensor.shape,"y크기")
    
    realvalues = prediction * (test_y_denominator + 1e-7) + np.min(testY, 0) 

    accuracy = (1 - (sum_mse/365)) * 100
    print('Accuracy:', accuracy.item())

plt.plot(testY_tensor)
plt.plot(prediction.data.numpy())
plt.legend(['original', 'prediction'])
plt.show()
