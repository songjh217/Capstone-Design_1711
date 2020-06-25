import torch
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch.nn.init 
import math
import numpy

data_parameter = 6
data_parameter_test = 5
hidden_week_dim = 7
hidden_weekend_dim = 6
output_dim = 1 
learning_rate = 0.01
iterations = 1000

#trainset가져오기
train_data = pd.read_csv('./trainset_for_capstone_2020.csv', usecols=['rainfall_all','maxtemp_all',
                        'avgtemp_all','sensible_all','GDP','max_power'])
trainxy_data = train_data.values
train_set = trainxy_data  #2014.1.1 ~ 2019.12.31 / size : (2191,6)

train_set_np = np.array(train_set)
change = np.reshape(train_set_np,(313,7,data_parameter)) # 2191 / 7 = 313 딱 나누어 떨어짐.(일주일씩 정렬 //토,일 분리해주려고)

# 수 목 금 토 일 월 화 [3],[4] >> 2014년 1월 1일은 수요일임
weekend = change[:, 3:5,:] # 토~일 자르기
#weekend 주말완성

sat = np.delete(change,3,axis=1) #토요일 자르기
week = np.delete(sat,3,axis=1) #일요일 자르기
#week 주간완성

WEEK = np.reshape(week,(1565,data_parameter)) #사이즈변형 (build_dataset들어가게 하려고)
WEEKEND = np.reshape(weekend,(626,data_parameter)) # 사이즈변형

#testset가져오기
test_data = pd.read_csv('./testset_2020_whole.csv', usecols=['rainfall_all','maxtemp_all', 
                        'avgtemp_all','sensible_all','GDP'])
testxy_data = test_data.values
test_set = testxy_data  # 2020년 6/23 ~ 8/31 / size : (70, 5)

test_set_np = np.array(test_set)
change_test = np.reshape(test_set_np,(53,7,data_parameter_test)) # 70 / 7 = 10으로 딱 나누어 떨어짐.(일주일씩 정렬 //토,일 분리해주려고)

# 화 수 목 금 토 일 월 [4],[5] >> 2020년 6월 23일은 화요일임
weekend_test = change_test[:, 2:4,:] # 토~일 자르기
print(weekend_test[-1])

sat = np.delete(change_test,2,axis=1) # 토요일 자르기
week_test = np.delete(sat,2,axis=1) # 일요일 자르기
print(week_test[0])

WEEK_test = np.reshape(week_test,(265,data_parameter_test)) #사이즈변형 (build_dataset들어가게 하려고)
WEEKEND_test = np.reshape(weekend_test,(106,data_parameter_test)) # 사이즈변형

#전력량 단위 Y_2019
Y_2019 = train_set[-365:-1,-1] #(2191,6) 중 뒤에서 364개(2019년 1/1 ~ 12/30  >> 7로 나눠떨어지게 하려고)
#2019년 1월 1일은 (화)요일부터 시작

Y_np = np.array(Y_2019)
change_Y = np.reshape(Y_np,(52,7)) # 364/7 = 52 (일주일씩 정렬 //토,일 분리해주려고)

weekend_Y = change_Y[:, 4:6] # 토~일 자르기
#weekend_Y 주말완성

sat_Y = np.delete(change_Y,4,axis=1) #토요일 자르기
week_Y = np.delete(sat_Y,4,axis=1) #일요일 자르기
#week_Y 주간완성

WEEK_Y = np.reshape(week_Y,(260,1)) #사이즈변형 (build_dataset들어가게 하려고)
WEEKEND_Y = np.reshape(weekend_Y,(104,1)) # 사이즈변형

device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.manual_seed(0)

class Power_week(torch.nn.Module): 
    def __init__(self, input_dim, hidden_dim, output_dim, layers):
        super(Power_week, self).__init__()
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
        self.fc_cnn = torch.nn.Linear(64, 5, bias=True)
        torch.nn.init.xavier_uniform_(self.fc_cnn.weight)

        self.LSTM = torch.nn.LSTM(input_dim, hidden_week_dim, num_layers=layers, batch_first=True) 
        self.fc_rnn = torch.nn.Linear(hidden_week_dim, output_dim, bias=True) 

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

class Power_weekend(torch.nn.Module): 
    def __init__(self, input_dim, hidden_dim, output_dim, layers):
        super(Power_weekend, self).__init__()
        self.layer1 = torch.nn.Sequential(
            torch.nn.Conv2d(1, 32, kernel_size=2, stride=1, padding=1), 
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=1))
        self.layer2 = torch.nn.Sequential(
            torch.nn.Conv2d(32, 128, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer3 = torch.nn.Sequential(
            torch.nn.Conv2d(128, 64, kernel_size=2, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2))
        self.fc_cnn = torch.nn.Linear(64, 2, bias=True)
        torch.nn.init.xavier_uniform_(self.fc_cnn.weight)

        self.LSTM = torch.nn.LSTM(input_dim, hidden_weekend_dim, num_layers=layers, batch_first=True) 
        self.fc_rnn = torch.nn.Linear(hidden_weekend_dim, output_dim, bias=True) 

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

#build_dataset(차원늘리기)
def build_dataset_week(time_series): # for 2014~2019 (1565,6)
  seq_length = 5
  dataX = []
  dataY = []
  for i in range(0, len(time_series)-seq_length): 
    _x = time_series[i:i + seq_length, :-1] #5개씩 6개의 요소중 -1(실제전력량)은 빼고
    #_x[0] : 2014.1.1~5 / 5일 씩 5개 요소 (5, 5)
    _y = time_series[i+seq_length,-1]
    #_y[0] : 2014.1.8 / 최대전력량 (1)

    dataX.append(_x)
    dataY.append(_y)
  
  return np.array(dataX), np.array(dataY) 

def build_dataset_weekend(time_series): # for 2014~2019 (626,6)
  seq_length = 2
  dataX = []
  dataY = []

  for i in range(0, len(time_series)-seq_length): 
    _x = time_series[i:i + seq_length, :-1] 
    _y = time_series[i+seq_length, -1] 
    
    dataX.append(_x)
    dataY.append(_y)
    
  return np.array(dataX), np.array(dataY) 

def build_dataset_week_test(time_series): # (50,5)
  seq_length = 5
  dataX = []
  dataY = []
  for i in range(0, len(time_series)-seq_length): 
    _x = time_series[i:i + seq_length, :] 
    
    dataX.append(_x)
    
  return np.array(dataX) 

def build_dataset_weekend_test(time_series): # (20,5)
  seq_length = 2
  dataX = []
  dataY = []
  for i in range(0, len(time_series)-seq_length): 
    _x = time_series[i:i + seq_length, :] 
    
    dataX.append(_x)
    
  return np.array(dataX)

trainX_week, trainY_week = build_dataset_week(WEEK)
trainX_weekend, trainY_weekend = build_dataset_weekend(WEEKEND)

testX_week = build_dataset_week_test(WEEK_test)
testX_weekend = build_dataset_weekend_test(WEEKEND_test)
print(testX_weekend)

WEEK_Y = build_dataset_week_test(week_Y)
WEEKEND_Y = build_dataset_weekend_test(weekend_Y)

train_x_week_numerator = trainX_week - np.min(trainX_week, 0) 
train_x_week_denominator = np.max(trainX_week, 0) - np.min(trainX_week, 0)
train_x_week_set = train_x_week_numerator / (train_x_week_denominator + 1e-7)
train_y_week_numerator = trainY_week - np.min(trainY_week, 0) 
train_y_week_denominator = np.max(trainY_week, 0) - np.min(trainY_week, 0)
train_y_week_set = train_y_week_numerator / (train_y_week_denominator + 1e-7)

train_x_weekend_numerator = trainX_weekend - np.min(trainX_weekend, 0) 
train_x_weekend_denominator = np.max(trainX_weekend, 0) - np.min(trainX_weekend, 0)
train_x_weekend_set = train_x_weekend_numerator / (train_x_weekend_denominator + 1e-7)
train_y_weekend_numerator = trainY_weekend - np.min(trainY_weekend, 0) 
train_y_weekend_denominator = np.max(trainY_weekend, 0) - np.min(trainY_weekend, 0)
train_y_weekend_set = train_y_weekend_numerator / (train_y_weekend_denominator + 1e-7)

test_x_week_numerator = testX_week - np.min(testX_week, 0) 
test_x_week_denominator = np.max(testX_week, 0) - np.min(testX_week, 0)
test_x_week_set = test_x_week_numerator / (test_x_week_denominator + 1e-7)
test_x_weekend_numerator = testX_weekend - np.min(testX_weekend, 0) 
test_x_weekend_denominator = np.max(testX_weekend, 0) - np.min(testX_weekend, 0)
test_x_weekend_set = test_x_weekend_numerator / (test_x_weekend_denominator + 1e-7)

Y_upper = WEEK_Y - np.min(WEEK_Y) 
Y_whole_week = np.max(WEEK_Y) - np.min(WEEK_Y)
Y_week_set = Y_upper / (Y_whole_week + 1e-7)

Y_upper_= WEEKEND_Y - np.min(WEEKEND_Y) 
Y_whole_weekend = np.max(WEEKEND_Y) - np.min(WEEKEND_Y)
Y_weekend_set = Y_upper_ / (Y_whole_weekend + 1e-7)

trainX_week_tensor = torch.FloatTensor(train_x_week_set) 
trainY_week_tensor = torch.FloatTensor(train_y_week_set)
trainX_weekend_tensor = torch.FloatTensor(train_x_weekend_set) 
trainY_weekend_tensor = torch.FloatTensor(train_y_weekend_set)
testX_week_tensor = torch.FloatTensor(test_x_week_set)
testX_weekend_tensor = torch.FloatTensor(test_x_weekend_set)

Power_week_prediction = Power_week(1, hidden_week_dim, output_dim, 1)
Power_weekend_prediction = Power_weekend(1, hidden_weekend_dim, output_dim, 1)

criterion = torch.nn.MSELoss() 

week_optimizer = optim.Adam(Power_week_prediction.parameters(), lr=learning_rate) 
weekend_optimizer = optim.Adam(Power_weekend_prediction.parameters(), lr=learning_rate)

for i in range(iterations+1): 
    X = trainX_week_tensor
    Y = trainY_week_tensor
    
    week_optimizer.zero_grad()
    outputs = Power_week_prediction(X)
    outputs = torch.squeeze(outputs,dim=1)

    loss = criterion(outputs, Y) 
    loss.backward() 
    week_optimizer.step()
    if i % 100 == 0 :
      print(i, "loss: ", loss.item()) 
first = outputs[-1]

for i in range(iterations+1): 
    X_ = trainX_weekend_tensor
    Y_ = trainY_weekend_tensor
    
    weekend_optimizer.zero_grad()
    outputs_ = Power_weekend_prediction(X_)
    outputs_ = torch.squeeze(outputs_,dim=1)

    loss_ = criterion(outputs_, Y_) 
    loss_.backward() 
    weekend_optimizer.step()
    if i % 100 == 0 :
      print(i, "loss: ", loss_.item()) 
      
with torch.no_grad():
    prediction_week = Power_week_prediction(testX_week_tensor)
    prediction_week = torch.squeeze(prediction_week,dim=-1)    
    prediction_powerV_week = prediction_week *(Y_whole_week + 1e-7) + np.min(WEEK_Y) 
    print(prediction_powerV_week[129:173])
    prediction_weekend = Power_weekend_prediction(testX_weekend_tensor)
    prediction_weekend = torch.squeeze(prediction_weekend,dim=-1)
    prediction_powerV_weekend = prediction_weekend *(Y_whole_weekend + 1e-7) + np.min(WEEKEND_Y) #실제 전력예측량(365)
    print(prediction_powerV_weekend[52:70],"asdfasdfasdfasdfasdfasdasdfsadsdasdf")
    
df3 = pd.DataFrame(prediction_powerV_week)
df3.to_excel('평일_pred.xlsx', index=False)
df6 = pd.DataFrame(prediction_powerV_weekend)
df6.to_excel('주말_pred.xlsx', index=False)
