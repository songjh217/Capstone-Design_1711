# Q. 코스트가 어느정도가 되어야 overfitting이 아닌지

import torch
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch.nn.init 
import math
import numpy

train_data = pd.read_csv('./train.csv', usecols=['rainfall_all',
                        'avgtemp_all','humidity_all','GDP','max_power'])
trainxy_data = train_data.values
train_set = trainxy_data [::-1] 

test_data = pd.read_csv('./test.csv', usecols=['rainfall_all', 
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
      
      ==================================================
      
       with torch.no_grad():
    prediction = power_prediction(testX_tensor)
    prediction_powerV = prediction *(test_y_denominator + 1e-7) + np.min(testY, 0)  
                                                                                    
    MSE = torch.nn.MSELoss()                                                        

    part_1 = prediction_powerV[:90]
    part_2 = prediction_powerV[90:181]
    part_3 = prediction_powerV[181:273]                                             
    part_4 = prediction_powerV[273:]

    Y_1 = testY[:90]
    Y_2 = testY[90:181]                                                             
    Y_3 = testY[181:273]
    Y_4 = testY[273:]

    Y_tensor_1 = testY_tensor[:90]
    Y_tensor_2 = testY_tensor[90:181]
    Y_tensor_3 = testY_tensor[181:273]                                              
    Y_tensor_4 = testY_tensor[273:]

    part_n_1 = prediction[:90]
    part_n_2 = prediction[90:181]                                                   
    part_n_3 = prediction[181:273]
    part_n_4 = prediction[273:]

    MSE_whole = MSE(testY_tensor,prediction)
    MSE_1 = MSE(Y_tensor_1,part_n_1) 
    MSE_2 = MSE(Y_tensor_2,part_n_2)
    MSE_3 = MSE(Y_tensor_3,part_n_3)
    MSE_4 = MSE(Y_tensor_4,part_n_4)                                                

    RMSE_whole = torch.sqrt(MSE_whole) 
    RMSE_1 = torch.sqrt(MSE_1)
    RMSE_2 = torch.sqrt(MSE_2)
    RMSE_3 = torch.sqrt(MSE_3)
    RMSE_4 = torch.sqrt(MSE_4)

    MAPE_whole =  np.mean(np.abs((testY - prediction_powerV.numpy()) / testY)) * 100
    MAPE_1 =  np.mean(np.abs((Y_1 - part_1.numpy()) / Y_1)) * 100
    MAPE_2 =  np.mean(np.abs((Y_2 - part_2.numpy()) / Y_2)) * 100
    MAPE_3 =  np.mean(np.abs((Y_3 - part_3.numpy()) / Y_3)) * 100
    MAPE_4 =  np.mean(np.abs((Y_4 - part_4.numpy()) / Y_4)) * 100

prediction_powerV_nn = prediction_powerV.numpy() 
y = testY_tensor.numpy() 

RMSE_whole = RMSE_whole.numpy()
RMSE_1 = RMSE_1.numpy()
RMSE_2 = RMSE_2.numpy()
RMSE_3 = RMSE_3.numpy()
RMSE_4 = RMSE_4.numpy()

print('전력사용량(Y) : ',testY)
print('전력예측값(prediction) : ',prediction_powerV_nn)
print()
print('RMSE(전체) : ',RMSE_whole)
print('MAPE(전체) : ',MAPE_whole)
print()
print('RMSE(1분기) : ',RMSE_1)
print('RMSE(2분기) : ',RMSE_2)
print('RMSE(3분기) : ',RMSE_3)
print('RMSE(4분기) : ',RMSE_4)
print()
print('MAPE(1분기) : ',MAPE_1)
print('MAPE(2분기) : ',MAPE_2)
print('MAPE(3분기) : ',MAPE_3)
print('MAPE(4분기) : ',MAPE_4)

plt.plot(testY)
plt.plot(prediction_powerV.data.numpy())
plt.legend(['original', 'prediction'])
plt.show()

===================================

print("1월")
plt.plot(testY[:31])
plt.plot(prediction_powerV[:31].data.numpy())
plt.legend(['original', 'prediction'])
plt.show()
==================================
print("2월")
plt.plot(testY[31:59])
plt.plot(prediction_powerV[31:59].data.numpy())
plt.legend(['original', 'prediction'])
plt.show()
===================================
print("3월")
plt.plot(testY[59:90])
plt.plot(prediction_powerV[59:90].data.numpy())
plt.legend(['original', 'prediction'])
plt.show()
==================================
print("4월")
plt.plot(testY[90:120])
plt.plot(prediction_powerV[90:120].data.numpy())
plt.legend(['original', 'prediction'])
plt.show()
===================================
print("5월")

plt.plot(testY[120:151])
plt.plot(prediction_powerV[120:151].data.numpy())
plt.legend(['original', 'prediction'])
plt.show()
===================================
print("6월")

plt.plot(testY[151:181])
plt.plot(prediction_powerV[151:181].data.numpy())
plt.legend(['original', 'prediction'])
plt.show()
=========================
print("7월")

plt.plot(testY[181:212])
plt.plot(prediction_powerV[181:212].data.numpy())
plt.legend(['original', 'prediction'])
plt.show()
============================
print("8월")

plt.plot(testY[212:243])
plt.plot(prediction_powerV[212:243].data.numpy())
plt.legend(['original', 'prediction'])
plt.show()
=================================
print("9월")

plt.plot(testY[243:273])
plt.plot(prediction_powerV[243:273].data.numpy())
plt.legend(['original', 'prediction'])
plt.show()
====================================
print("10월")

plt.plot(testY[273:304])
plt.plot(prediction_powerV[273:304].data.numpy())
plt.legend(['original', 'prediction'])
plt.show()
=====================================
print("11월")

plt.plot(testY[304:334])
plt.plot(prediction_powerV[304:334].data.numpy())
plt.legend(['original', 'prediction'])
plt.show()
======================================
print("12월")

plt.plot(testY[334:])
plt.plot(prediction_powerV[334:].data.numpy())
plt.legend(['original', 'prediction'])
plt.show()


