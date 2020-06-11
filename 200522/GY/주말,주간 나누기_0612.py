'''
문제점
1. 2014년 1월 1일은 월요일이 아니다 >> 수요일이다
2. 이를 해결하기 위해서 부족한 부분은 컷하거나 채워 넣는다
(앞부분은 컷, 뒷부분은 채워 넣는게 나을 듯)
3. trainSet은 데이터처리 끝났는데 testSet도 해줘야 한다.
'''

import torch
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch.nn.init 
import math
import numpy

train_data = pd.read_csv('./train_heat.csv', usecols=['rainfall_all',
                        'avgtemp_all','humidity_all','GDP','max_power'])
trainxy_data = train_data.values
train_set = trainxy_data
#print(train_set.shape) # 1827  :  1/1 추가해서 딱 떨어지게 변경


train_set_np = np.array(train_set)
change = np.reshape(train_set_np,(261,7,5)) #1827x5 >> 261x7x5 로 변경
#print(change.shape) # (261, 7, 5)
#print(change[0]) # 2014/1/1~2014/1/7 


week = change[:, :5,:] # 월~금 자르기
WEEK = np.reshape(week,(1305,5)) #사이즈변형 


weekend = change[:, 5:,:] # 토~일 자르기
WEEKEND = np.reshape(weekend,(522,5)) # 사이즈변형

#======================================================================================
test_data = pd.read_csv('./test_heat.csv', usecols=['rainfall_all', 
                        'avgtemp_all','humidity_all','GDP','max_power'])
testxy_data = test_data.values
test_set = testxy_data  # (365, 5)

test_set_np = np.array(test_set)

change = np.reshape(test_set_np,(261,7,5)) #1827x5 >> 261x7x5 로 변경
#print(change.shape) # (261, 7, 5)
#print(change[0]) # 2014/1/1~2014/1/7 


week = change[:, :5,:] # 월~금 자르기
WEEK = np.reshape(week,(1305,5)) #사이즈변형 


weekend = change[:, 5:,:] # 토~일 자르기
WEEKEND = np.reshape(weekend,(522,5)) # 사이즈변형


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
            torch.nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1), 
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = torch.nn.Sequential(
            torch.nn.Conv2d(32, 128, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU())
            # torch.nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer3 = torch.nn.Sequential(
            torch.nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
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
        out = self.layer3(out)
        out = out.view(out.size(0), -1) 
        out = self.fc_cnn(out)
        out = torch.unsqueeze(out,-1)
        x, _status = self.LSTM(out) 
        x = self.fc_rnn(x[:, -1])
        return x

power_prediction = Power(1, hidden_dim, output_dim, 1)

criterion = torch.nn.MSELoss() 

optimizer = optim.Adam(power_prediction.parameters(), lr=learning_rate) 


def build_dataset_week(time_series):
  #print(time_series.shape,"지금") # (1305,5)
  seq_length = 5
  dataX = []
  dataY = []
  for i in range(0, len(time_series)-seq_length-1): 
    _x = time_series[i:i + seq_length, :-1] #(5,4)
    #_x[0] : 2014.1.1~5 / 네개요소
    _y = time_series[i+seq_length,-1]
    #_y[0] : 2014.1.8 / 최대전력량

    
    dataX.append(_x)
    dataY.append(_y)
    
  
  return np.array(dataX), np.array(dataY) 

def build_dataset_weekend(time_series):
  seq_length = 2
  dataX = []
  dataY = []

  for i in range(0, len(time_series)-seq_length): # 520번
    _x = time_series[i:i + seq_length, :-1] 
    #print(_x.shape,"_x") #(2,4) : 2014.1.6~7    >>잘 안되면 (4,4)로 바꾸기
    #print(_x[0],"_x[0]")
    _y = time_series[i+seq_length, -1] 
    #print(_y.size,"_y") #[0] : 2014.1.13
    
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
#print(trainX_week.shape,"train_week") # 1300,5,4
#1301이 아니라 1300인 이유 : train_X보다 train_Y는 그 다음날의 값이어야 해서 Y에 1305의 값이 들어가야 하기 때문에 X는 1300~1304까지들어갈 수 밖에 없음 
#print(trainY_week.shape,"train_week") # 1300 : 2014.1.8 ~ 2018.1.1



trainX_weekend, trainY_weekend = build_dataset_weekend(WEEKEND)
# print(trainX_weekend.shape,"train_weekend_X") # 520, 2, 4
# print(trainY_weekend.shape,"train_weekend_Y") # 520


#test도 다시.......

testX, testY = build_dataset(test_set, seq_length)
# testset : 2019년 > 358일
print(testX.shape,"testX") # 358,7,4
print(testY.shape,"testY") # 358,1

'''
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

    loss = criterion(outputs, Y) # 실제오차보다 너무 작아서 MSE > RMSE로 일단 표시 #실제 오차 아님.
    loss.backward() 
    optimizer.step()
    if i % 100 == 0 :
      print(i, "loss: ", loss.item()) 



'''

