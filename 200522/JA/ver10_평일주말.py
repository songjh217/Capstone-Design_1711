
import torch
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch.nn.init 
import math
import numpy

train_data = pd.read_csv('/content/drive/My Drive/Colab Notebooks/캡스톤/data/train_heat.csv', usecols=['rainfall_all',
                        'avgtemp_all','humidity_all','GDP','max_power'])
trainxy_data = train_data.values
train_set = trainxy_data


#print(train_set.shape,"첫") # 1/1 추가해서 1827만들어줌. 딱 떨어지게
train_set_np = np.array(train_set)
train_set_np = np.append(train_set_np, np.array([0,0,0,0,0]))
#print(train_set_np.shape)
change = np.reshape(train_set_np,(261,7,5))
#print(change)
#print(change.shape,"바뀜") # (261, 7, 5)
#print(change[0])
#2014/1/1~2014/1/7 
week = change[:, :5,:]
WEEK = np.reshape(week,(1305,5))
#print(WEEK.shape,"다시 합쳐")
weekend = change[:, 5:,:]
WEEKEND = np.reshape(weekend,(522,5))
#print(WEEKEND.shape,"다시 합쳐")
#print(week.shape,"week")
#print(weekend.shape,"weekend")
#print("===============================")
#print(week[0])
#print(weekend[0])
#print("===============================")



test_data = pd.read_csv('/content/drive/My Drive/Colab Notebooks/캡스톤/data/test_heat.csv', usecols=['rainfall_all', 
                        'avgtemp_all','humidity_all','GDP','max_power'])
testxy_data = test_data.values
test_set = testxy_data 

device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.manual_seed(0)




seq_length = 7 
data_parameter = 5
hidden_dim = 10 
output_dim = 1 
learning_rate = 0.01
iterations = 1000


class Power(torch.nn.Module): 
    def __init__(self, input_dim, hidden_dim, output_dim, layers):
        super(Power, self).__init__()
        self.layer1 = torch.nn.Sequential(
            torch.nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1), 
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = torch.nn.Sequential(
            torch.nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU())
            # torch.nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer3 = torch.nn.Sequential(
            torch.nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2))
        self.fc_cnn = torch.nn.Linear(64, 5, bias=True)
        torch.nn.init.xavier_uniform_(self.fc_cnn.weight)

        self.LSTM = torch.nn.LSTM(input_dim, hidden_dim, num_layers=layers, batch_first=True) 
        self.fc_rnn = torch.nn.Linear(hidden_dim, output_dim, bias=True) 

    def forward(self, x):
        x = torch.unsqueeze(x,1)
        w,h = x.shape[2],x.shape[3] 
        out = self.layer1(x) 
        out = self.layer2(out)
        out = self.layer3(out)
        out = out.view(out.size(0), -1) 
        out = self.fc_cnn(out)
        out = torch.unsqueeze(out,-1)
        x, _status = self.LSTM(out) 
        x = self.fc_rnn(x[:, -1])
        return x

class weekend_Power(torch.nn.Module): 
    def __init__(self, input_dim, hidden_dim, output_dim, layers):
        super(weekend_Power, self).__init__()
        self.layer1 = torch.nn.Sequential(
            torch.nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1), 
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = torch.nn.Sequential(
            torch.nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU())
            # torch.nn.MaxPool2d(kernel_size=2, stride=2))
        '''self.layer3 = torch.nn.Sequential(
            torch.nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2))'''
        self.fc_cnn = torch.nn.Linear(128, 2, bias=True)
        torch.nn.init.xavier_uniform_(self.fc_cnn.weight)

        self.LSTM = torch.nn.LSTM(input_dim, hidden_dim, num_layers=layers, batch_first=True) 
        self.fc_rnn = torch.nn.Linear(hidden_dim, output_dim, bias=True) 

    def forward(self, x):
        x = torch.unsqueeze(x,1)
        w,h = x.shape[2],x.shape[3] 
        out = self.layer1(x) 
        out = self.layer2(out)
        #out = self.layer3(out)
        out = out.view(out.size(0), -1) 
        out = self.fc_cnn(out)
        out = torch.unsqueeze(out,-1)
        x, _status = self.LSTM(out) 
        x = self.fc_rnn(x[:, -1])
        return x

power_prediction = Power(1, hidden_dim, output_dim, 1)

weekend_power_prediction = weekend_Power(1, hidden_dim, output_dim, 1)

criterion = torch.nn.MSELoss() 

optimizer = optim.Adam(power_prediction.parameters(), lr=learning_rate) 

weekend_optimizer = optim.Adam(weekend_power_prediction.parameters(), lr=learning_rate)

def build_dataset_week(time_series):
  print(time_series.shape,"지금")
  seq_length = 5
  dataX = []
  dataY = []
  for i in range(0, len(time_series)-seq_length): 
    _x = time_series[i:i + seq_length, :-1] 
    _y = time_series[i+seq_length, [-1]] 
    
    dataX.append(_x)
    dataY.append(_y)
    
  
  return np.array(dataX), np.array(dataY) 

def build_dataset_weekend(time_series):
  seq_length = 2
  dataX = []
  dataY = []
  for i in range(0, len(time_series)-seq_length): 
    _x = time_series[i:i + seq_length, :-1] 
    _y = time_series[i+seq_length, [-1]] 
    
    dataX.append(_x)
    dataY.append(_y)
    
  return np.array(dataX), np.array(dataY) 


def build_dataset(time_series, seq_length):
  dataX = []
  dataY = []
  for i in range(0, len(time_series)-seq_length): 
    _x = time_series[i:i + seq_length, :-1] 
    _y = time_series[i+seq_length, [-1]] 
    
    dataX.append(_x)
    dataY.append(_y)
    
  return np.array(dataX), np.array(dataY) 

trainX_week, trainY_week = build_dataset_week(WEEK)
print(trainX_week.shape,"train_week") # 1300,5,4
trainX_weekend, trainY_weekend = build_dataset_weekend(WEEKEND)
print(trainX_weekend.shape,"train_weekend") # 520, 2, 4
testX, testY = build_dataset(test_set, seq_length) 

trainX_week_numerator = trainX_week - np.min(trainX_week, 0) 
trainX_week_denominator = np.max(trainX_week, 0) - np.min(trainX_week, 0)
trainX_week_set = trainX_week_numerator / (trainX_week_denominator + 1e-7)

trainY_week_numerator = trainY_week - np.min(trainY_week, 0) 
trainY_week_denominator = np.max(trainY_week, 0) - np.min(trainY_week, 0)
trainY_week_set = trainY_week_numerator / (trainY_week_denominator + 1e-7)

trainX_weekend_numerator = trainX_weekend - np.min(trainX_weekend, 0) 
trainX_weekend_denominator = np.max(trainX_weekend, 0) - np.min(trainX_weekend, 0)
trainX_weekend_set = trainX_weekend_numerator / (trainX_weekend_denominator + 1e-7)

trainY_weekend_numerator = trainY_weekend - np.min(trainY_weekend, 0) 
trainY_weekend_denominator = np.max(trainY_week, 0) - np.min(trainY_weekend, 0)
trainY_weekend_set = trainY_weekend_numerator / (trainY_weekend_denominator + 1e-7)



test_x_numerator = testX - np.min(testX, 0) 
test_x_denominator = np.max(testX, 0) - np.min(testX, 0)
test_x_set = test_x_numerator / (test_x_denominator + 1e-7)

test_y_numerator = testY - np.min(testY, 0) 
test_y_denominator = np.max(testY, 0) - np.min(testY, 0)
test_y_set = test_y_numerator / (test_y_denominator + 1e-7)


trainX_week_tensor = torch.FloatTensor(trainX_week_set) 
trainY_week_tensor = torch.FloatTensor(trainY_week_set)

trainX_weekend_tensor = torch.FloatTensor(trainX_weekend_set) 
trainY_weekend_tensor = torch.FloatTensor(trainY_weekend_set)


testX_tensor = torch.FloatTensor(test_x_set)
testY_tensor = torch.FloatTensor(test_y_set)

criterion = torch.nn.MSELoss() 

optimizer = optim.Adam(power_prediction.parameters(), lr=learning_rate) 

for i in range(iterations+1): 
    week_X = trainX_week_tensor
    week_Y = trainY_week_tensor

    optimizer.zero_grad()
    outputs = power_prediction(week_X)

    loss = criterion(outputs, week_Y) # 실제오차보다 너무 작아서 MSE > RMSE로 일단 표시 #실제 오차 아님.
    loss.backward() 
    optimizer.step()
    if i % 100 == 0 :
      print(i, "loss: ", loss.item()) 

for i in range(iterations+1): 
    weekend_X = trainX_weekend_tensor
    weekend_Y = trainY_weekend_tensor
    
    weekend_optimizer.zero_grad()
    outputs = weekend_power_prediction(weekend_X)

    loss = criterion(outputs, weekend_Y) # 실제오차보다 너무 작아서 MSE > RMSE로 일단 표시 #실제 오차 아님.
    loss.backward() 
    optimizer.step()
    if i % 100 == 0 :
      print(i, "loss: ", loss.item()) 




