import torch
import torch.nn as nn
import torch.optim as optim
import numpy
import matplotlib.pyplot as plt
import pandas as pd
import torch.nn.functional as F
from google.colab import drive

device = 'cuda' if torch.cuda.is_available() else 'cpu'

torch.manual_seed(0)

seq_length = 7 
data_parameter = 2
hidden_dim = 10 
output_dim = 1 
learning_rate = 0.01
iterations = 1200



train_data = pd.read_csv('/content/drive/My Drive/Colab Notebooks/캡스톤/data/train.csv', usecols=['rainfall_all','max_power'])#,'avgtemp_all','humidity_all','GDP','max_power'])
trainxy_data = train_data.values
train_set = trainxy_data # (1826, 5)


test_data = pd.read_csv('/content/drive/My Drive/Colab Notebooks/캡스톤/data/test.csv', usecols=['rainfall_all','max_power'])#,'avgtemp_all','humidity_all','GDP','max_power'])
testxy_data = test_data.values
test_set = testxy_data #[365, 5]

def build_dataset(time_series, seq_length):
  # print(time_series.shape,"d") # (177, 5) / (83, 5)
  dataX = []
  dataY = []
  for i in range(0, len(time_series)-seq_length): 
    _x = time_series[i:i + seq_length, :-1] 
    # print(_x.shape,"7 4") #[7, 5]
    _y = time_series[i+seq_length, [-1]]  #1이 아니라 7이 나와야하는거 아닌가(이건 따로 나뒀을때 : x)
    # print(_y.shape,"aa")#1 > 7 1 
    
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


class Power(torch.nn.Module):
  def __init__(self, input_dim, hidden_dim, output_dim, layers):
    super(Power, self).__init__()
    self.linear1 = torch.nn.Linear(1, 256, bias=True)
    self.linear2 = torch.nn.Linear(256,128, bias=True)
    self.linear3 = linear3 = torch.nn.Linear(128, 1, bias=True)
    self.relu = torch.nn.ReLU()

    self.rnn = torch.nn.LSTM(input_dim, hidden_dim, num_layers=layers, batch_first=True)
    self.fc = torch.nn.Linear(hidden_dim, output_dim, bias=True)
    
  def forward(self, x):
    out = self.linear1(x)
    
    out = self.relu(out)
    out = self.linear2(out)
    out = self.linear3(out)
    
    x, _status = self.rnn(out)
    #print(_status)
    x = self.fc(x[:, -1])
    return x
  
power_prediction = Power(1, hidden_dim, output_dim, 1)

criterion = torch.nn.MSELoss() 
optimizer = optim.Adam(power_prediction.parameters(), lr=learning_rate)

for i in range(iterations+1): 
    X = trainX_tensor # [170, 7, 5]
    Y = trainY_tensor # [170, 1] > 170 7 1
    # Y = Y.squeeze() # Y는 [170 7]
    
    optimizer.zero_grad()
    outputs = power_prediction(X) #1819 1

    # print(outputs.shape,"output") #1819 1
    # print(Y.shape,"YYYYYYYYYYY") #1819

    loss = criterion(outputs, Y) 
    loss.backward() 
    optimizer.step()
    if i % 100 == 0 :
      print(i, "loss: ", loss.item()) #, "prediction: ", outputs, "true Y: ", Y) 






with torch.no_grad():
    prediction = power_prediction(testX_tensor)
    prediction_powerV = prediction *(test_y_denominator + 1e-7) + np.min(testY, 0) #실제 전력예측량(365일)
    

    

    #RMSE, MAPE는 분기별
    MSE = torch.nn.MSELoss() #RMSE
    p = torch.nn.MSELoss()
    # print(p.shape)

    prediction = power_prediction(testX_tensor)
    prediction_powerV = prediction *(test_y_denominator + 1e-7) + np.min(testY, 0) 
    
    #RMSE, MAPE는 분기별
    MSE = torch.nn.MSELoss() 

    #예측값 4분기
    part_1 = prediction_powerV[:90]
    part_2 = prediction_powerV[90:181]
    part_3 = prediction_powerV[181:273]
    part_4 = prediction_powerV[273:]
    
    #예측값_정규화
    part_n_1 = prediction[:90]
    part_n_2 = prediction[90:181]
    part_n_3 = prediction[181:273]
    part_n_4 = prediction[273:]


    #실제값 4분기
    Y_1 = testY[:90]
    Y_2 = testY[90:181]
    Y_3 = testY[181:273]
    Y_4 = testY[273:]
    
    #실제값_정규화
    Y_tensor_1 = testY_tensor[:90]
    Y_tensor_2 = testY_tensor[90:181]
    Y_tensor_3 = testY_tensor[181:273]
    Y_tensor_4 = testY_tensor[273:]
    

    MSE_whole = MSE(testY_tensor,prediction)
    MSE_1 = MSE(Y_tensor_1,part_n_1) # 실제:실제값
    MSE_2 = MSE(Y_tensor_2,part_n_2)
    MSE_3 = MSE(Y_tensor_3,part_n_3)
    MSE_4 = MSE(Y_tensor_4,part_n_4)


    
    RMSE_whole = torch.sqrt(MSE_whole) #나눠준 후 루트
    RMSE_1 = torch.sqrt(MSE_1)
    RMSE_2 = torch.sqrt(MSE_2)
    RMSE_3 = torch.sqrt(MSE_3)
    RMSE_4 = torch.sqrt(MSE_4)



    MAPE_whole =  np.mean(np.abs((testY - prediction_powerV.numpy()) / testY)) * 100
    MAPE_1 =  np.mean(np.abs((Y_1 - part_1.numpy()) / Y_1)) * 100
    MAPE_2 =  np.mean(np.abs((Y_2 - part_2.numpy()) / Y_2)) * 100
    MAPE_3 =  np.mean(np.abs((Y_3 - part_3.numpy()) / Y_3)) * 100
    MAPE_4 =  np.mean(np.abs((Y_4 - part_4.numpy()) / Y_4)) * 100

#여기서부터는 변수 액셀에 적용하면 굳굳
prediction_powerV_nn = prediction_powerV.numpy() #예측값
y = testY #실제
print(len(prediction_powerV_nn))
print(len(y))
print(len(test_set))


#RMSE 텐서에서 넘파이 형태로
RMSE_whole = RMSE_whole.numpy()
RMSE_1 = RMSE_1.numpy()
RMSE_2 = RMSE_2.numpy()
RMSE_3 = RMSE_3.numpy()
RMSE_4 = RMSE_4.numpy()


print('우리예측')
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

#plt.plot(kpx_testY)
plt.plot(testY)
plt.plot(prediction_powerV.data.numpy())
plt.legend(['original', 'prediction'])
plt.show()

print('1분기')
#plt.plot(kpx_testY[:90])
plt.plot(testY[:90])
plt.plot(prediction_powerV[:90].data.numpy())
plt.legend(['original', 'prediction'])
plt.show()

print('2분기')
#plt.plot(kpx_testY[90:181])
plt.plot(testY[90:181])
plt.plot(prediction_powerV[90:181].data.numpy())
plt.legend(['original', 'prediction'])
plt.show()

print('3분기')
#plt.plot(kpx_testY[181:273])
plt.plot(testY[181:273])
plt.plot(prediction_powerV[181:273].data.numpy())
plt.legend(['original', 'prediction'])
plt.show()

print('4분기')
#plt.plot(kpx_testY[273:])
plt.plot(testY[273:])
plt.plot(prediction_powerV[273:].data.numpy())
plt.legend(['original', 'prediction'])
plt.show()
