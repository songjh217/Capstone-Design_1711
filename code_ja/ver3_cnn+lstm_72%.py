import torch
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import torch.nn.init #for cnn


device = 'cuda' if torch.cuda.is_available() else 'cpu'

torch.manual_seed(0)


class Power(torch.nn.Module): 
    

    def __init__(self, input_dim, hidden_dim, output_dim, layers):
        super(Power, self).__init__()
        self.layer1 = torch.nn.Sequential(
            torch.nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1), # (입력,출력,커널사이즈,스트라이드,패딩)
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


seq_length = 7 
data_parameter = 5
hidden_dim = 10 
output_dim = 1 
learning_rate = 0.01
iterations = 500

train_data = pd.read_csv('/content/drive/My Drive/Colab Notebooks/캡스톤/data/train.csv', usecols=['rainfall_all','avgtemp_all','humidity_all','GDP','max_power'])
trainxy_data = train_data.values
train_set = trainxy_data [::-1] # (1826, 5)


test_data = pd.read_csv('/content/drive/My Drive/Colab Notebooks/캡스톤/data/test.csv', usecols=['rainfall_all','avgtemp_all','humidity_all','GDP','max_power'])
testxy_data = test_data.values
test_set = trainxy_data [::-1] #[365, 5]
'''

def minmax_scaler(data):
  numerator = data - np.min(data, 0) 
  denominator = np.max(data, 0) - np.min(data, 0)


  return numerator / (denominator + 1e-7), denominator
'''
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


'''
train_set, total_train_V = minmax_scaler(train_set1)
test_set, total_test_V  = minmax_scaler(test_set1) 
print(total_test_V)
'''

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
    error = torch.nn.MSELoss(reduction='sum')
    sum_mse = error(prediction, testY_tensor)

    #되돌리기
    realvalues = prediction * (test_y_denominator + 1e-7) + np.min(testY, 0) 
    print(realvalues)
    print(testY)

    accuracy = (1 - (sum_mse/76)) * 100
    #정확도 어떤 방식으로 내는지 논문 확인*******************************
    # correct_prediction = torch.argmax(prediction, 1) == testY
    
    # print(accuracy)
    print('Accuracy:', accuracy.item())



'''
    prediction = power_prediction(testX_tensor)
 

    #되돌리기
    realvalues = prediction * (test_y_denominator + 1e-7) + np.min(testY, 0) 
    print(realvalues)
    print(testY)

    error = torch.nn.MSELoss(reduction='sum')
    sum_mse = error(realvalues, torch.FloatTensor(testY))
    accuracy = (1 - (sum_mse/76)) * 100

    print('Accuracy:', accuracy.item())


    #correct_prediction = torch.argmax(prediction, 1) == testY_tensor
    #accuracy = correct_prediction.float().mean()
    #print('Accuracy:', accuracy.item())

'''

    #정확도 어떤 방식으로 내는지 논문 확인*******************************
    # correct_prediction = torch.argmax(prediction, 1) == testY
    
    # print(accuracy)

a = prediction[:10]
b = testY_tensor[:10] #torch써서 tensor형으로 진행 / 그래프랑 같은거 확인함.
cat = torch.cat([a, b], dim=1)
experiment = a - b
print(cat) #[a, b]로 10x2


# print(experiment)
plt.plot(testY_tensor)
plt.plot(prediction.data.numpy())
plt.legend(['original', 'prediction'])
plt.show()
