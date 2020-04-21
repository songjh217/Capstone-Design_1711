#우리데이터 아님. / 정확도 방식 확인해봐야함. / 
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

al_data = pd.read_csv('./GOOG.cvs.csv', usecols=['Open','High','Low','Close','Volume'])
xy_data = al_data.values

xy = xy_data [::-1]

train_size = int(len(xy) * 0.7)
train_set = xy[0:train_size] 
test_set = xy[train_size - seq_length:] 


def minmax_scaler(data):
  numerator = data - np.min(data, 0) 
  denominator = np.max(data, 0) - np.min(data, 0)
  return numerator / (denominator + 1e-7) 

def build_dataset(time_series, seq_length):
  dataX = []
  dataY = []
  for i in range(0, len(time_series)-seq_length): 
    _x = time_series[i:i + seq_length, :] 
    
    _y = time_series[i+seq_length, [-1]] 
    
    dataX.append(_x)
    dataY.append(_y)
    
  return np.array(dataX), np.array(dataY) 


train_set = minmax_scaler(train_set)
test_set = minmax_scaler(test_set) 

trainX, trainY = build_dataset(train_set, seq_length)
testX, testY = build_dataset(test_set, seq_length) 


trainX_tensor = torch.FloatTensor(trainX) 
trainY_tensor = torch.FloatTensor(trainY)


testX_tensor = torch.FloatTensor(testX)
testY_tensor = torch.FloatTensor(testY)




power_prediction = Power(1, hidden_dim, output_dim, 1)

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
      print(i, "loss: ", loss.item()) #, "prediction: ", outputs, "true Y: ", Y) 


with torch.no_grad():
    prediction = power_prediction(testX_tensor)
    error = torch.nn.MSELoss(reduction='sum')
    sum_mse = error(prediction, testY_tensor)
    accuracy = (1 - (sum_mse/76)) * 100
    #정확도 어떤 방식으로 내는지 논문 확인*******************************
    # correct_prediction = torch.argmax(prediction, 1) == testY
    
    # print(accuracy)
    print('Accuracy:', accuracy.item())

a = prediction[:10]
b = testY_tensor[:10] #torch써서 tensor형으로 진행 / 그래프랑 같은거 확인함.
cat = torch.cat([a, b], dim=1)
experiment = a - b
print(cat) #[a, b]로 10x2


# print(experiment)
plt.plot(testY)
plt.plot(prediction.data.numpy())
plt.legend(['original', 'prediction'])
plt.show()


