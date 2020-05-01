#cnn-lstm-loss:(cnn+lstm)을 학습시킨것 / 지금 모델은 cnn에 들어가는 데이터자체가 lstm에 들어가는 데이터 기준으로 나눈 것 
# cnn에 들어가는 x자체에 7일간의 y데이터가 들어가 있는 형태 : 이부분이 제일 이상함.
#cnn에 y데이터 제외한 1819.7.4가 들어가서 학습시켜야하고 cnn이미지 학습법처럼 학습시킨다는 것은 7x5를 한장처럼보고 학습시키겟다는 건데 그때의 y는 뭐가 될건데...? (이미지학습을 이미지 학습이 아닌 다른 곳에서 사용할때 어떻게 사용하는지에 대한 공부부족)
#그래서 이렇게 7x4를 7로만들거라면 (공부가 부족해서 섞이면 안된다고 생각이 들음) 그냥 4개를 통해 1을 예측하는 예측모델과 무엇이 다른가.logistic+lstm??
#결론 : 이미지데이터가 아닌 데이터에서 cnn을 통해 특징을 뽑아낼때 어떤 방식으로 이루어지는지에 대한 공부가 필요함.
#     : (추가) logistic + lstm 해서 정확도 구해보기.(비교대상)

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

train_data = pd.read_csv('./train.csv', usecols=['rainfall_all','avgtemp_all','humidity_all','GDP','max_power'])
trainxy_data = train_data.values
train_set = trainxy_data [::-1] # (1826, 5)


test_data = pd.read_csv('./test.csv', usecols=['rainfall_all','avgtemp_all','humidity_all','GDP','max_power'])
testxy_data = test_data.values
test_set = testxy_data [::-1] #[365, 5]


def minmax_scaler(data):
  numerator = data - np.min(data, 0) 
  denominator = np.max(data, 0) - np.min(data, 0)
  return numerator / (denominator + 1e-7) 

def build_dataset(time_series, seq_length):
  # print(time_series.shape,"d") # (177, 5) / (83, 5)
  dataX = []
  dataY = []
  for i in range(0, len(time_series)-seq_length): 
    _x = time_series[i:i + seq_length, :] 
    # print(_x.shape,"7 4") #[7, 5]
    _y = time_series[i+seq_length, [-1]]  #1이 아니라 7이 나와야하는거 아닌가(이건 따로 나뒀을때 : x)
    # print(_y.shape,"aa")#1 > 7 1 
    
    dataX.append(_x)
    dataY.append(_y)
    
  return np.array(dataX), np.array(dataY) 


train_set = minmax_scaler(train_set)
test_set = minmax_scaler(test_set) 

trainX, trainY = build_dataset(train_set, seq_length)
# print(trainX.shape,"x")#170 7 5 #177>170이 됨 : 1826>1819가 되는 건가
# print(trainY.shape,"y") #170 1 #이게 170 7이 돼야한다고 생각
#(170, 7, 5) x
#(170, 7, 1) y
testX, testY = build_dataset(test_set, seq_length) 


trainX_tensor = torch.FloatTensor(trainX) 
trainY_tensor = torch.FloatTensor(trainY)


testX_tensor = torch.FloatTensor(testX)
testY_tensor = torch.FloatTensor(testY)




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


