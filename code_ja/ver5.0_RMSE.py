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
    #예측값 4분기
    part_1 = prediction[:90]
    part_2 = prediction[90:181]
    part_3 = prediction[181:273]
    part_4 = prediction[273:]

    #실제값 4분기
    Y_1 = testY[:90]
    Y_2 = testY[90:181]
    Y_3 = testY[181:273]
    Y_4 = testY[273:]


    MSE_whole = MSE(testY_tensor,prediction)
    MSE_1 = MSE(torch.FloatTensor(Y_1),part_1)
    MSE_2 = MSE(torch.FloatTensor(Y_2),part_2)
    MSE_3 = MSE(torch.FloatTensor(Y_3),part_3)
    MSE_4 = MSE(torch.FloatTensor(Y_4),part_4)

    
    
    RMSE_whole = torch.sqrt(MSE_whole)

    MAPE_whole = torch.mean(abs((testY_tensor - prediction) / testY_tensor)) * 100
    MAPE_1 = torch.mean(abs((torch.FloatTensor(Y_1) - part_1) / torch.FloatTensor(Y_1))) * 100
    MAPE_2 = torch.mean(abs((torch.FloatTensor(Y_2) - part_2) / torch.FloatTensor(Y_2))) * 100
    MAPE_3 = torch.mean(abs((torch.FloatTensor(Y_3) - part_3) / torch.FloatTensor(Y_3))) * 100
    MAPE_4 = torch.mean(abs((torch.FloatTensor(Y_4) - part_4) / torch.FloatTensor(Y_4))) * 100

   


plt.plot(testY)
plt.plot(prediction_powerV.data.numpy())
plt.legend(['original', 'prediction'])
plt.show()

#넘파이로 변환
prediction_powerV = prediction_powerV.numpy() #예측값
y = testY_tensor.numpy() #실제값
RMSE_whole = RMSE_whole.numpy()
