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










'''

capstone_data = pd.read_csv('./capstone.csv', usecols=['rainfall_all','maxtemp_all',
                        'avgtemp_all','sensible_all','GDP','max_power'])
capstonexy_data = capstone_data.values
capstone_set = capstonexy_data 
answer = capstone_set[3:,-1]
print(answer[0]) #122 > 122/7 = 17...3 >>>>>119
#answer_np = np.array(answer)
#change = np.reshape(answer_np,(313,7,data_parameter))

'''

















train_data = pd.read_csv('./trainset_for_capstone_2020.csv', usecols=['rainfall_all','maxtemp_all',
                        'avgtemp_all','sensible_all','GDP','max_power'])
trainxy_data = train_data.values
train_set = trainxy_data 



#정규화풀어줄 Y_min구하기
Y_2019 = train_set[-365:-1,-1] #(2191,6) : 2191 / 7 = 313으로 딱 떨어짐 (가공할 필요없음)
#Y_2019(화)
Y_np = np.array(Y_2019)
change_Y = np.reshape(Y_np,(52,7))
#print(change_Y[0])
weekend_Y = change_Y[:, 4:6] # 토~일 자르기
#print(weekend_Y[0])

sat_Y = np.delete(change_Y,4,axis=1)
week_Y = np.delete(sat_Y,4,axis=1)#수~화
#print(week_Y[0])




train_set_np = np.array(train_set)
change = np.reshape(train_set_np,(313,7,data_parameter)) # 2184 x 6 >> 312 x 7 x 6 로 변경
#print(change.shape) 

#print(change[0]) # 2014/1/1~2014/1/7 : 수요일

# 수 목 금 토 일 월 화 [3],[4]
weekend = change[:, 3:5,:] # 토~일 자르기


sat = np.delete(change,3,axis=1)
week = np.delete(sat,3,axis=1)#수~화


#print(week[0],"수~화인지 확인(4,5주말제외)")
#print(weekend[0],"토 일인지 확인") # ㅇㅇ 토일맞음

#print(week.shape)
#print(weekend.shape)


WEEK = np.reshape(week,(1565,data_parameter)) #사이즈변형 
WEEKEND = np.reshape(weekend,(626,data_parameter)) # 사이즈변형

#======================================================================================
test_data = pd.read_csv('./testset_for_capstone_2020.csv', usecols=['rainfall_all','maxtemp_all', 
                        'avgtemp_all','sensible_all','GDP'])
testxy_data = test_data.values
test_set = testxy_data  # (70, 5) : 딱 6/23~8/31까지

#print(test_set.shape) # (70, 5) : 70/7 = 10

test_set_np = np.array(test_set)

change_test = np.reshape(test_set_np,(10,7,data_parameter_test)) # 10x7x5 로 변경
#print(change_test[0]) # 2020/6/23~2020/6/29 : 화~월 

weekend_test = change_test[:, 4:6,:] # 토~일 자르기

sat = np.delete(change_test,4,axis=1)
week_test = np.delete(sat,4,axis=1)#수~화


#print(week_test[0],"수~화인지 확인(4,5주말제외)")
#print(weekend_test[0],"토 일인지 확인") # ㅇㅇ 토일맞음

#print(week_test.shape)

WEEK_test = np.reshape(week_test,(50,data_parameter_test)) #사이즈변형 
WEEKEND_test = np.reshape(weekend_test,(20,data_parameter_test)) # 사이즈변형
#=================================================



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
        super(Power_weekend, self).__init__() #****************************
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

def build_dataset_week_test(time_series):
  seq_length = 5
  dataX = []
  dataY = []
  for i in range(0, len(time_series)-seq_length): 
    _x = time_series[i:i + seq_length, :] 
    
    dataX.append(_x)
    
  return np.array(dataX) 

def build_dataset_weekend_test(time_series):
  seq_length = 2
  dataX = []
  dataY = []
  for i in range(0, len(time_series)-seq_length): 
    _x = time_series[i:i + seq_length, :] 
    
    dataX.append(_x)
    
  return np.array(dataX)


trainX_week, trainY_week = build_dataset_week(WEEK)


trainX_weekend, trainY_weekend = build_dataset_weekend(WEEKEND)
#print(trainY_week.shape,"adfasdf")
#print(trainY_weekend.shape,"adfasdf")
#testX_week = build_dataset_week_test(WEEK_test)
#testX_weekend = build_dataset_weekend_test(WEEKEND_test)
#print(WEEK_test.shape,"aaaaaaaaaaaa") # 50 5 주간(일주일씩 뽑은거)
#print(testX_week.shape,"fffffffff") # 45 5

WEEK_Y = build_dataset_week_test(week_Y)
WEEKEND_Y = build_dataset_weekend_test(weekend_Y)
print(weekend_Y.shape)
#==================================================================

train_x_week_numerator = trainX_week - np.min(trainX_week, 0) #**************************사이즈 몇인지 확인 int면 그냥 계산 아니면 build거쳐서 넣기
train_x_week_denominator = np.max(trainX_week, 0) - np.min(trainX_week, 0)

print(np.max(trainX_week, 0).shape,"adsfasdfasdfasdfas")
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













Y_upper = WEEK_Y - np.min(WEEK_Y) #*******************사이즈 몇인지 확인 int면 그냥 계산 아니면 build거쳐서 넣기
Y_whole_week = np.max(WEEK_Y) - np.min(WEEK_Y)
Y_week_set = Y_upper / (Y_whole_week + 1e-7)

'''
test_y_week_numerator = testY_week - np.min(testY_week, 0) 
test_y_week_denominator = np.max(testY_week, 0) - np.min(testY_week, 0)
test_y_week_set = test_y_week_numerator / (test_y_week_denominator + 1e-7)
'''



Y_upper_= WEEKEND_Y - np.min(WEEKEND_Y) #**************************사이즈 몇인지 확인 int면 그냥 계산 아니면 build거쳐서 넣기
Y_whole_weekend = np.max(WEEKEND_Y) - np.min(WEEKEND_Y)
Y_weekend_set = Y_upper_ / (Y_whole_weekend + 1e-7)










test_x_weekend_numerator = testX_weekend - np.min(testX_weekend, 0) 
test_x_weekend_denominator = np.max(testX_weekend, 0) - np.min(testX_weekend, 0)
test_x_weekend_set = test_x_weekend_numerator / (test_x_weekend_denominator + 1e-7)
'''
test_y_weekend_numerator = testY_weekend - np.min(testY_weekend, 0) 
test_y_weekend_denominator = np.max(testY_weekend, 0) - np.min(testY_weekend, 0)
test_y_weekend_set = test_y_weekend_numerator / (test_y_weekend_denominator + 1e-7)
'''

trainX_week_tensor = torch.FloatTensor(train_x_week_set) 
trainY_week_tensor = torch.FloatTensor(train_y_week_set)

trainX_weekend_tensor = torch.FloatTensor(train_x_weekend_set) 
trainY_weekend_tensor = torch.FloatTensor(train_y_weekend_set)

testX_week_tensor = torch.FloatTensor(test_x_week_set)
#testY_week_tensor = torch.FloatTensor(test_y_week_set)

testX_weekend_tensor = torch.FloatTensor(test_x_weekend_set)
#testY_weekend_tensor = torch.FloatTensor(test_y_weekend_set)


#testY_tensor = torch.FloatTensor(testY) ***************이거 뭐야 왜있는거야 없애기


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

    loss = criterion(outputs, Y) # 실제오차보다 너무 작아서 MSE > RMSE로 일단 표시 #실제 오차 아님.
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

    loss_ = criterion(outputs_, Y_) # 실제오차보다 너무 작아서 MSE > RMSE로 일단 표시 #실제 오차 아님.
    loss_.backward() 
    weekend_optimizer.step()
    if i % 100 == 0 :
      print(i, "loss: ", loss_.item()) 



with torch.no_grad():
    prediction_week = Power_week_prediction(testX_week_tensor)
    prediction_week = torch.squeeze(prediction_week,dim=-1)    
    prediction_powerV_week = prediction_week *(Y_whole_week + 1e-7) + np.min(week_Y) #실제 전력예측량(365)
    #First = first *(test_y_week_denominator + 1e-7) + np.min(testY_week, 0)
    #6/30~8/31까지 예측 (63)
    print(prediction_powerV_week.shape)
    
    
    print(prediction_powerV_week)
    #print(torch.FloatTensor(answer))

    prediction_weekend = Power_weekend_prediction(testX_weekend_tensor)
    prediction_weekend = torch.squeeze(prediction_weekend,dim=-1)
    prediction_powerV_weekend = prediction_weekend *(Y_whole_weekend + 1e-7) + np.min(weekend_Y) #실제 전력예측량(365)
    print(prediction_powerV_weekend)
    ################Y_whole_weekend곱해주는 값이 하나 instance 라서 안됨
    print(prediction_powerV_weekend.shape)
    




print('============평일============')
plt.plot(prediction_powerV_week.data.numpy())
plt.legend(['prediction'])
plt.show()
print('============주말============')
#plt.plot(testY_weekend)
plt.plot(prediction_powerV_weekend.data.numpy())
plt.legend(['prediction'])
plt.show()

'''
df1 = pd.DataFrame([[MAPE_week_whole],[RMSE_week_whole]])
df2 = pd.DataFrame(testY_week) 
df3 = pd.DataFrame(prediction_powerV_week)
df1.to_excel('평일_W.xlsx', index=False)
df2.to_excel('평일_y.xlsx', index=False)
df3.to_excel('평일_pred.xlsx', index=False)

df4 = pd.DataFrame([[MAPE_weekend_whole],[RMSE_weekend_whole]])
df5 = pd.DataFrame(testY_weekend) 
df6 = pd.DataFrame(prediction_powerV_weekend)
df4.to_excel('주말_W.xlsx', index=False)
df5.to_excel('주말_y.xlsx', index=False)
df6.to_excel('주말_pred.xlsx', index=False)
'''
