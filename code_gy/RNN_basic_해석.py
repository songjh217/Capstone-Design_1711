#RNN_for_capstone_완성.ipynb
import torch
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


torch.manual_seed(0)

seq_length = 7 # 7일
data_parameter = 5 # 5차원 데이터
hidden_dim = 10 
output_dim = 1 # FC의 차원
learning_rate = 0.01
iterations = 1500 # 500번 돌리겠다.(batch말고 epochs)

#xy = np.loadtxt("./GOOG.cvs.csv", delimiter=",") #데이터 로드
al_data = pd.read_csv('./GOOG.cvs.csv', usecols=['Open','High','Low','Close','Volume'])
xy_data = al_data.values
# print(xy_data) # 20190327부터
xy = xy_data [::-1] #데이터를 역순으로 배열
# print(xy) # 20200326부터
# print(xy.shape) #(253, 5) # 토,일은 제외

train_size = int(len(xy) * 0.7) # 253*0.7 = 177
# print(xy[0])
train_set = xy[0:train_size] # 177일치를 trainset으로 쓰겠다
test_set = xy[train_size - seq_length:] # 171일부터 253일까지를 test로 쓰겠다
#빼주는 이유는 단위가 7일씩 넣을 거라서


#최대 최소값을 찾아서 linear하게 나눠주는 방식의 스케일링 (0과 1사이로 보내줌.)
def minmax_scaler(data): # train,test_set 넣어줌 train_set:(177,5)
  # print(np.min(data, 0))
  numerator = data - np.min(data, 0) #177.5 데이터를 0차원중 최소값을 뽑아서 빼라 : 한마디로 가장 작은 값을 0으로하겠다.????? 굳이 왜하는지...
  denominator = np.max(data, 0) - np.min(data, 0) # 모든 데이터중 최대 - 최소값.
  return numerator / (denominator + 1e-7) # 1e-7하는 이유는? 1이 안되게 하려고 하는 거 같은데.... 왜지?
  # 일단 이렇게 하면 return 값은 확실히 0~1사이로 정리됨.

#레이블과 학습 대상인 값들을 나눠줌
def build_dataset(time_series, seq_length): # 0~1로 표현된 데이터들이 거치는 함수
  dataX = []
  dataY = []
  for i in range(0, len(time_series)-seq_length): #170번 반복 //참고:들어온 data의 size는 177
    _x = time_series[i:i + seq_length, :] #(7,5) //뒤거 빼도됨.:걍 7일차데이터 다가져오는 것
    # print(_x[-1,-1]) # 마지막중에 마지막 하나 
    
    _y = time_series[i+seq_length, [-1]] # x 다음 하루치.(그러니까 x input으로 7일치가 들어가면 하루치를 예측하는데 이게 그다음날 하루 따라서 예측하는 값의 실제 값은 그다음날 이걸 y_라고 함.)
    #그중 맨 마지막 요소값(volume인거 같음)

    # print(_x,"->", _y) #input -> output실제값.
    
    dataX.append(_x)
    dataY.append(_y)
    
  return np.array(dataX), np.array(dataY) # 각각 170개

# scaling data >> 줄이기 위해 하는 짓
train_set = minmax_scaler(train_set) # 왜 이렇게 하는지는 모르겠지만 0~1값으로 표현됨.
test_set = minmax_scaler(test_set) # 동일

# make train-test dataset to input
trainX, trainY = build_dataset(train_set, seq_length) #트레이닝 데이터를 7일씩 학습데이터로 만들어주겠다(전처리느낌)
testX, testY = build_dataset(test_set, seq_length) #마찬가지 : 형태 넘파이

# convert to tensor
trainX_tensor = torch.FloatTensor(trainX) # np를 tensor로 7일씩들어가는 170개 데이터
trainY_tensor = torch.FloatTensor(trainY)

testX_tensor = torch.FloatTensor(testX)
testY_tensor = torch.FloatTensor(testY)

print(trainY_tensor)
print(testY_tensor)

#---------------------------------------------------------------------------------------------------
class Net(torch.nn.Module):
  def __init__(self, input_dim, hidden_dim, output_dim, layers): #inputsize는 
    super(Net, self).__init__() #torch.nn.Module의 __init__ 동일하게 사용
    self.LSTM = torch.nn.LSTM(input_dim, hidden_dim, num_layers=layers, batch_first=True) #LSTM사용 중 #어떻게 되고 있는지 보려면 Module뜯어봐야함.
    self.fc = torch.nn.Linear(hidden_dim, output_dim, bias=True) #마지막 output dimention맞춰주려고 1로 (10>>1)
    
  def forward(self, x): #net()에 데이터 들어오면 실행 (7일씩 170개 tensor형 데이터)
    # print(x.shape,"그냥 입력데이터") #torch.Size([170, 7, 5]) 
    x, _status = self.LSTM(x) #LSTM 썼으니까 바꿔줌 
    # print(x.shape,"LSTM통과한 output") #torch.Size([170, 7, 10]) : hidden_dim을 10으로 해서
    # print(x.shape)
    # print(x[:, -1].shape)
    x = self.fc(x[:, -1])#x의 전체중 맨 마지막부분만 가져오기.... ????왜 이러는데????
    #shape는 10>>1로 감
    return x
#---------------------------------------------------------------------------------------------------


net = Net(data_parameter, hidden_dim, output_dim, 1) #Net이라는 class사용하겠다 선언 (5,10,1,1)
#net을 통과해 나오는 return 값은 x로 1차원 output나옴.

criterion = torch.nn.MSELoss() #loss 선언
optimizer = optim.Adam(net.parameters(), lr=learning_rate) # optimizer 선언

# real 학습시작
for i in range(iterations): #500번 반복
    optimizer.zero_grad()
    outputs = net(trainX_tensor)  # 1차원 형태 outputs /input은 7일씩들어가는 170개 데이터
    #만든 net모델에 train데이터 넣음 (170,7,5) -> (170,1)
    print(trainX_tensor.shape,"이거야")
    print(outputs.shape, "바뀜")
    loss = criterion(outputs, trainY_tensor) #예측값,실제값
    loss.backward() #(backpropagation) > decent
    optimizer.step() #update
    # print(i, loss.item()) #학습진행할 수록  loss 내려가는거 볼 수 있음.
    # print(outputs[1],trainY_tensor[1])

# plt.plot(trainY_tensor)
# plt.plot(outputs.data.numpy())
# plt.legend(['real', 'practice'])
# plt.show()
#왜 이부분에서
# torch.Size([76, 7, 10])
# torch.Size([76, 1]) 이게 되는거지?

# print(testY, "testY")
plt.plot(testY)
plt.plot(trainY_tensor)
plt.legend(['test', 'train'])
# a = -1* net(testX_tensor)
# plt.plot(a.data.numpy())
# plt.legend(['original', 'prediction'])
plt.show()
