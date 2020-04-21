import torch
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import torch.nn.init


torch.manual_seed(0)

seq_length = 1 #1일
input_dim = 1 #5차원 데이터
hidden_dim = 10 
output_dim = 1 #FC의 차원
learning_rate = 0.01
iterations = 500
layers = 1

#xy = np.loadtxt("/content/drive/My Drive/Colab Notebooks/Dataset/GOOG.cvs.csv", delimiter=",") #데이터 로드
al_data = pd.read_csv('/content/drive/My Drive/Colab Notebooks/Dataset/GOOG.cvs.csv', usecols=['Open','High','Low','Volume','Close'])
xy_data = al_data.values

xy = xy_data [::-1] #데이터를 역순으로 배열

'''
x_data = xy_data[:,0:-1]
y_data = xy_data[:,[-1]]

x = torch.FloatTensor(x_data)
y = torch.FloatTensor(y_data)
print(x)
'''

train_size = int(len(xy) * 0.7) 
train_set = xy[0:train_size]
test_set = xy[train_size - seq_length:]

#print(train_set)

#최대 최소값을 찾아서 linear하게 나눠주는 방식의 스케일링
def minmax_scaler(data):
  numerator = data - np.min(data, 0)
  denominator = np.max(data, 0) - np.min(data, 0)
  return numerator / (denominator + 1e-7)

#레이블과 학습 대상인 값들을 나눠줌
def build_dataset(time_series, seq_length):
  dataX = []
  dataY = []
  for i in range(0, len(time_series)-seq_length):
    _x = time_series[i:i + seq_length, :]
    _y = time_series[i+seq_length, [-1]]
    # print(_x,"->", _y)
    dataX.append(_x)
    dataY.append(_y)
  return np.array(dataX), np.array(dataY)

# scaling data
train_set = minmax_scaler(train_set)
test_set = minmax_scaler(test_set)
'''
# make train-test dataset to input
trainX, trainY = build_dataset(train_set, seq_length)
testX, testY = build_dataset(test_set, seq_length)
'''
trainX = train_set[:,0:-1]
trainY = train_set[:,[-1]]
testX = test_set[:,0:-1]
testY = test_set[:,[-1]]


# convert to tensor
trainX_tensor = torch.FloatTensor(trainX)
trainX_tensor = torch.unsqueeze(trainX_tensor, 1)
trainX_tensor = torch.unsqueeze(trainX_tensor, 1)
trainY_tensor = torch.FloatTensor(trainY)

testX_tensor = torch.FloatTensor(testX)
testX_tensor = torch.unsqueeze(testX_tensor, 1)
testX_tensor = torch.unsqueeze(testX_tensor, 1)
testY_tensor = torch.FloatTensor(testY)

class Net(torch.nn.Module):
  def __init__(self):
    super(Net, self).__init__()
    self.Cnn()

  def Cnn(self):
    self.layer1 = torch.nn.Sequential(
        torch.nn.Conv2d(1, 1, kernel_size=1, stride=1, padding=0),
        torch.nn.ReLU(),
        torch.nn.MaxPool2d(kernel_size=1, stride=1))
    self.rnn = torch.nn.LSTM(input_dim, hidden_dim, num_layers=layers, batch_first=True)
    self.fc = torch.nn.Linear(hidden_dim, output_dim, bias=True)
    
  
  def forward(self, x):
    x = self.layer1(x)
    x = x.view(-1,4,1) 
    #print(x)
    x, _status = self.rnn(x)
   # print(_status)
    x = self.fc(x[:, -1])
    return x
    
    model = Net()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
criterion = torch.nn.MSELoss()
trainX_tensor.size()
trainY_tensor.size()

trainX_tensor.size()
for epoch in range(1000):
    hypothesis = model(trainX_tensor)
    cost = criterion(hypothesis,trainY_tensor)

    # cost로 H(x) 개선
    optimizer.zero_grad()
    cost.backward()
    optimizer.step()


    # 10번마다 로그 출력
    if epoch % 10 == 0:
        print('Epoch {:4d} Cost: {:.6f}'.format(
            epoch, cost.item()
        ))

plt.plot(testY)
plt.plot(Net(testX_tensor).data.numpy())
plt.legend(['original', 'prediction'])
plt.show()
 
