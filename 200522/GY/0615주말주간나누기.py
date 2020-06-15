
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
train_set = trainxy_data #1827 : 2019.1.1 추가해서 떨어지게 변경


train_set_np = np.array(train_set)
change = np.reshape(train_set_np,(261,7,5)) #1827x5 >> 261x7x5 로 변경
#print(change.shape) # (261, 7, 5)
#print(change[0]) # 2014/1/1~2014/1/7 : 수요일


# 수 목 금 토 일 월 화 [3],[4]
weekend = change[:, 3:5,:] # 토~일 자르기

sat = np.delete(change,3,axis=1)
week = np.delete(sat,3,axis=1)#수~화


#print(week[0],"수~화인지 확인(4,5주말제외)")
#print(weekend[0],"토 일인지 확인") # ㅇㅇ 토일맞음

WEEK = np.reshape(week,(1305,5)) #사이즈변형 
WEEKEND = np.reshape(weekend,(522,5)) # 사이즈변형

#======================================================================================
test_data = pd.read_csv('./test_heat.csv', usecols=['rainfall_all', 
                        'avgtemp_all','humidity_all','GDP','max_power'])
testxy_data = test_data.values
test_set = testxy_data  # (365, 5) > 365 / 7 = 52 ...1

test_set_np = np.array(test_set)

change_test = np.reshape(test_set_np,(52,7,5)) #364x5 >> 52x7x5 로 변경
#print(change_test[0]) # 2014/1/1~2014/1/7 : 화~월 

weekend_test = change_test[:, 4:6,:] # 토~일 자르기

sat = np.delete(change_test,4,axis=1)
week_test = np.delete(sat,4,axis=1)#수~화


#print(week_test[0],"수~화인지 확인(4,5주말제외)")
#print(weekend_test[0],"토 일인지 확인") # ㅇㅇ 토일맞음

WEEK_test = np.reshape(week_test,(260,5)) #사이즈변형 
WEEKEND_test = np.reshape(weekend_test,(104,5)) # 사이즈변형
#=================================================





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
  for i in range(0, len(time_series)-seq_length): 
    _x = time_series[i:i + seq_length, :-1] #(5,4)
    #_x[0] : 2014.1.1~5 / 네개요소
    _y = time_series[i+seq_length,-1]
    #_y[0] : 2014.1.8 / 최대전력량

#이렇게 그냥 학습시키면 12/25~12/31까지 학습시켜서 1/1예측한 거 까지 학습시키고 끝나는 형태임.
#심지어 test는 2019.1.8부터 예측됨.(카톡사진) : (1/2~7은 예측 안시키고 있음)1/2예측하려면 12/26(수)부터 값 들어가야 함.*********(수정)
    
    dataX.append(_x)
    dataY.append(_y)
    
  
  return np.array(dataX), np.array(dataY) 

def build_dataset_weekend(time_series):
  #print(time_series[0],"지금") #522 5
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
#print(trainX_week.shape,"train_week") # 1300,5,4(y때문에 이렇게 됨. 데이터 수정해서 고쳐야 됨(수정***))
#1301이 아니라 1300인 이유 : train_X보다 train_Y는 그 다음날의 값이어야 해서 Y에 1305까지의 값이 들어가야 하기 때문에 X는 1300~1304까지들어갈 수 밖에 없음 
#print(trainY_week.shape,"train_week") # 1300 : 2014.1.8 ~ 2018.1.1
#print(trainX_week[-1],"train_week")
#print(trainY_week[-1],"Y예측의 실제값 몇일인지 확인")


trainX_weekend, trainY_weekend = build_dataset_weekend(WEEKEND)
#print(trainX_weekend[0],"train_weekend_X") # 520, 2, 4
#print(trainY_weekend[0],"train_weekend_Y") # 520
#2014.1.4~5 >> 2014.1.11


#test도 다시.......
testX_week, testY_week = build_dataset_week(WEEK_test)
testX_weekend, testY_weekend = build_dataset_weekend(WEEKEND_test)
# testset : 2019년 > 357일
#12/23이 마지막 데이터 : 12/23~27(월) 들어가서 12월30일 예측(y)라서 357


#2019.1.1~12.30(31일 빼면 364일)
#print(testX_week.shape,"testX") # 255.5.4 /1월 1일,2일,3일,4일,7일 >> 8일
#print(testX_weekend.shape,"testY") # 102.2.4 / 1월 5일,6일 >> 12일


#==================================================================

train_x_week_numerator = trainX_week - np.min(trainX_week, 0) 
train_x_week_denominator = np.max(trainX_week, 0) - np.min(trainX_week, 0)
train_x_week_set = train_x_week_numerator / (train_x_week_denominator + 1e-7)

train_y_week_numerator = trainY_week - np.min(trainY_week, 0) 
train_y_week_denominator = np.max(trainY_week, 0) - np.min(trainY_week, 0)
train_y_week_set = train_y_week_numerator / (train_y_week_denominator + 1e-7)


train_x_weekend_numerator = trainX_week - np.min(trainX_weekend, 0) 
train_x_weekend_denominator = np.max(trainX_weekend, 0) - np.min(trainX_weekend, 0)
train_x_weekend_set = train_x_weekend_numerator / (train_x_weekend_denominator + 1e-7)

train_y_weekend_numerator = trainY_weekend - np.min(trainY_weekend, 0) 
train_y_weekend_denominator = np.max(trainY_weekend, 0) - np.min(trainY_weekend, 0)
train_y_weekend_set = train_y_weekend_numerator / (train_y_weekend_denominator + 1e-7)

test_x_week_numerator = testX_week - np.min(testX_week, 0) 
test_x_week_denominator = np.max(testX_week, 0) - np.min(testX_week, 0)
test_x_week_set = test_x_week_numerator / (test_x_week_denominator + 1e-7)

test_y_week_numerator = testY_week - np.min(testY_week, 0) 
test_y_week_denominator = np.max(testY_week, 0) - np.min(testY_week, 0)
test_y_week_set = test_y_week_numerator / (test_y_week_denominator + 1e-7)

test_x_weekend_numerator = testX_weekend - np.min(testX_weekend, 0) 
test_x_weekend_denominator = np.max(testX_weekend, 0) - np.min(testX_weekend, 0)
test_x_weekend_set = test_x_weekend_numerator / (test_x_weekend_denominator + 1e-7)

test_y_weekend_numerator = testY_weekend - np.min(testY_weekend, 0) 
test_y_weekend_denominator = np.max(testY_weekend, 0) - np.min(testY_weekend, 0)
test_y_weekend_set = test_y_weekend_numerator / (test_y_weekend_denominator + 1e-7)


trainX_week_tensor = torch.FloatTensor(train_x_week_set) 
trainY_week_tensor = torch.FloatTensor(train_y_week_set)

trainX_weekend_tensor = torch.FloatTensor(train_x_weekend_set) 
trainY_weekend_tensor = torch.FloatTensor(train_y_weekend_set)

testX_week_tensor = torch.FloatTensor(test_x_week_set)
testY_week_tensor = torch.FloatTensor(test_y_week_set)

testX_weekend_tensor = torch.FloatTensor(test_x_weekend_set)
testY_weekend_tensor = torch.FloatTensor(test_y_weekend_set)

criterion = torch.nn.MSELoss() 

optimizer = optim.Adam(power_prediction.parameters(), lr=learning_rate) 
#==================================================================================
for i in range(iterations+1): 
    X = trainX_week_tensor
    Y = trainY_week_tensor
    
    optimizer.zero_grad()
    outputs = power_prediction(X)

    loss = criterion(outputs, Y) # 실제오차보다 너무 작아서 MSE > RMSE로 일단 표시 #실제 오차 아님.
    loss.backward() 
    optimizer.step()
    if i % 100 == 0 :
      print(i, "loss: ", loss.item()) 

for i in range(iterations+1): 
    X_ = trainX_weekend_tensor
    Y_ = trainY_weekend_tensor
    
    optimizer.zero_grad()
    outputs_ = power_prediction(X)

    loss_ = criterion(outputs_, Y_) # 실제오차보다 너무 작아서 MSE > RMSE로 일단 표시 #실제 오차 아님.
    loss_.backward() 
    optimizer.step()
    if i % 100 == 0 :
      print(i, "loss: ", loss_.item()) 


