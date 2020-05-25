import torch
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch.nn.init 
import math
import numpy

train_data = pd.read_csv('/content/drive/My Drive/Colab Notebooks/캡스톤/data/train.csv', usecols=['rainfall_all',
                        'avgtemp_all','humidity_all','GDP','max_power'])

trainxy_data = train_data.values
train_set = trainxy_data [::-1] 

test_data = pd.read_csv('/content/drive/My Drive/Colab Notebooks/캡스톤/data/test.csv', usecols=['rainfall_all',
                        'avgtemp_all','humidity_all','GDP','max_power'])
testxy_data = test_data.values
test_set = testxy_data [::-1]  

test_kpx_data = pd.read_csv('/content/drive/My Drive/Colab Notebooks/캡스톤/data/kpx_land_power_2019.csv', usecols=['MAX_all'])
test_kpx_y_data = test_kpx_data.values
test_kpx_set = test_kpx_y_data [::-1]  


device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.manual_seed(0)

seq_length = 7 
data_parameter = 5
hidden_dim = 10 
output_dim = 1 
learning_rate = 0.01
iterations = 1200


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





def build_dataset(time_series, seq_length):
  dataX = []
  dataY = []
  for i in range(0, len(time_series)-seq_length): 
    _x = time_series[i:i + seq_length, :-1] 
    _y = time_series[i+seq_length, [-1]] 
    
    dataX.append(_x)
    dataY.append(_y)
    
  return np.array(dataX), np.array(dataY) 

def build_kpx_dataset(time_series, seq_length):
  dataY = []
  for i in range(0, len(time_series)-seq_length): 
    _y = time_series[i+seq_length, [-1]] 
    
    dataY.append(_y)
    
  return np.array(dataY) 

trainX, trainY = build_dataset(train_set, seq_length)
testX, testY = build_dataset(test_set, seq_length) 

kpx_testY = build_kpx_dataset(test_kpx_set, seq_length)


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

with torch.no_grad():
    prediction = power_prediction(testX_tensor)
    prediction_powerV = prediction *(test_y_denominator + 1e-7) + np.min(testY, 0) #실제 전력예측량(365일)

    #RMSE, MAPE는 분기별
    MSE = torch.nn.MSELoss() #RMSE
    p = torch.nn.MSELoss()
    # print(p.shape)

    RMSE = []
    MAPE = []
    RMSE_W = torch.sqrt(MSE_whole).item()
    MAPE_W =  np.mean(np.abs((testY - prediction_powerV.numpy()) / testY)) * 100

    for i in range(len(prediction)):
      MSE_whole = MSE(testY_tensor[i],prediction[i])
      RMSE_whole = torch.sqrt(MSE_whole)
      RMSE.append(RMSE_whole.item())
      #print('일 :',i+1,'RMSE :',RMSE_whole.numpy())

    for i in range(len(prediction)):
      MAPE_whole =  np.mean(np.abs((testY[i] - prediction_powerV[i].numpy()) / testY[i])) * 100
      MAPE.append(MAPE_whole)
      #print('일 :',i+1,"  ",'MAPE :',MAPE_whole)

#RMSE_W, MAPE_W는 전체 한개로 나타낸거 총 1개 데이터
#RMSE, MAPE는 일별로 나타낸거 총 358개 데이터
#엑셀로 넘겨주면 되는 변수야! 

prediction_powerV_nn = prediction_powerV.numpy() #예측값
y = testY #실제값

#확인차 출력한거 굳이 너 돌릴때는 출력 안해도됨!
print(prediction_powerV_nn)
print(y)
print(RMSE_W) 
print(MAPE_W)
print(RMSE)
print(MAPE)
