import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch.nn.functional as F
from google.colab import drive

device = 'cuda' if torch.cuda.is_available() else 'cpu'

torch.manual_seed(0)
train_data = pd.read_csv('./train.csv', usecols=['rainfall_all','avgtemp_all','humidity_all','GDP','max_power'])
trainxy_data = train_data.values
train_set = trainxy_data [::-1] # (1826, 5)
train_set_tensor = torch.FloatTensor(train_set)

test_data = pd.read_csv('./test.csv', usecols=['rainfall_all','avgtemp_all','humidity_all','GDP','max_power'])
testxy_data = test_data.values
test_set = testxy_data [::-1] #[365, 5]

print(train_set)
print(train_set.shape)

W = torch.zeros((4,1),requires_grad=True) # 1x1:이래도 돼?  torch.zeros((3,1))해야 하는거 아닌가
b = torch.zeros((1,1),requires_grad=True) # 0. :초기화를 0으로 잡고 돌려보겠다.

print(W.shape)
print(b.shape)
# optimizer 설정
optimizer = optim.SGD([W, b], lr=0.00005005)
#SGD vs GD : GD는 전체를 다보고 레이트조정, SGD는 mini-batch만 보고 조정(발자국 작음)

#learning_rate = 0.01 #SGD버전이랑 같게 하려고
nb_epochs = 1000 
for epoch in range(nb_epochs + 1): # 1001번 돌리기 == update를 1001번해주기 같은 데이터로(나눠서 올리는게 아닌가봄) == 1001번 학습한다?

    # H(x) 계산
    W_tensor = torch.tensor(W)
    hypothesis = train_set.matmul(W_tensor) + b # [25X3]*[3x1]+[1x1] = [25x1] 인걸 보면 알아서 맞춰지는 듯

    # cost 계산
    cost = torch.mean((hypothesis - train_test) ** 2)
    #print(cost)

    # cost로 H(x) 개선
    optimizer.zero_grad() #초기화 매학습에 0으로 초기화 뭐를?
    cost.backward()
    optimizer.step() # rate 조정된 다음 W로

    # 100번마다 로그 출력
    if epoch % 100 == 0:
        print('Epoch {:4d}/{} Cost: {:.6f}'.format(epoch, nb_epochs,cost.item()))
