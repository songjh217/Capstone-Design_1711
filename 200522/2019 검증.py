# 0. 코드 작성에 필요한 라이브러리 불러오기
import torch
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch.nn.init 
import math
import numpy

device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.manual_seed(0)

# (하이퍼 파라미터지정)
data_parameter = 6      # 예측요소 수
hidden_week_dim = 5     # 주간 hidden_state 수(LSTM)
hidden_weekend_dim = 2  # 주말 hidden_state 수(LSTM)
output_dim = 1          # 출력(전력사용량) size 
learning_rate = 0.01    # learning rate (수정보폭)
iterations = 1000       # epoch 수 (반복학습 횟수)
# 1100, 1200에폭에서 가장 loss가 잘 떨어졌습니다.

# 1. 데이터 불러오기 (+ 데이터 변환_input_data_만들기)
# *(trainset데이터)
train_data = pd.read_csv('./jung_train.csv', usecols=['rainfall_all','maxtemp_all','avgtemp_all','sensible_all','GDP','max_power']) 
                        # pandas를 이용해서 jung_train라는 이름의 csv파일을 콤마(,)를 기준으로 불러온다.('jung_train.csv' : 2014~2018년도 데이터들이 정리된 파일)
                        # 이때, 엑셀에서 rainfall_all(강수량),maxtemp_all(최고기온),avgtemp_all(평균기온),sensible_all(체감온도),GDP,max_power(실제최대전력량)이란
                        #                                                                                                          이름의 열의 데이터만 불러온다.
                        # 예측요소는 6개
train_set = train_data.values # 읽은 값을 list형태로 변경해줌(.values)
# size : (1827(날짜), 6(요소 수))
# [365(2014년) + 365(2015년) + 366(2016년) + 365(2017년) + 365(2018년) + 1(2019.1.1//일주일단위(7)로 사이즈 맞추기 위해서) = 1827]

train_set_np = np.array(train_set) # 계산을 쉽게 하기 위해 numpy를 사용 > numpy array형태로 변환
change = np.reshape(train_set_np,(261,7,data_parameter)) #(1827x6) >> (261x7(일주일)x6)로 차원변경 (요일별로 정렬해줌)
# 차원변경하는 이유 : 주간/주말을 나누기 위해서
# ex) change[0] >> 2014/1/1~2014/1/7 이 들어가 있으며 2014.1.1은 수요일이다.
'''
change에는 데이터가 이런식으로 들어가 있다.
수 목 금 토 일 월 화
1  2  3  4  5  6  7
8  9  10 ...해서 2019년 1월 1일 데이터까지
'''

# 1-1. 주말/주간 나눠주기
# trainset은 '수 목 금 토 일 월 화' 순서대로 저장되어 있음. 

#(주말나누기)
# [3]번째 : 토요일, [4]번째 : 일요일
weekend = change[:, 3:5,:] # 토~일 자르기
#weekend에는 '토 일'만 들어가 있음.

#(주간나누기)
sat = np.delete(change,3,axis=1) # [3]번째(토요일) 삭제
week = np.delete(sat,3,axis=1)   # [4]번째(일요일) 삭제 
#week에는 '수 목 금 월 화'만 남게 됨.

# 주말/주간 나눠준 데이터를 다시 2차원으로 변형
WEEK = np.reshape(week,(1305,data_parameter))      # (261X5,6)
WEEKEND = np.reshape(weekend,(522,data_parameter)) # (261x2,6)

# 1. 데이터 불러오기 (+ 데이터 변환_input_data_만들기)
# *(testset데이터)
test_data = pd.read_csv('./2019_testset.csv', usecols=['rainfall_all','maxtemp_all', 'avgtemp_all','sensible_all','GDP','max_power'])
                        # pandas를 이용해서 jung_test라는 이름의 csv파일을 콤마(,)를 기준으로 불러온다.('jung_test.csv' : 2019년도 데이터들이 정리된 파일)
                        # 이때, 엑셀에서 rainfall_all(강수량),maxtemp_all(최고기온),avgtemp_all(평균기온),sensible_all(체감온도),GDP,max_power(실제최대전력량)이란
                        #                                                                                                          이름의 열의 데이터만 불러온다.
                        # 예측요소는 6개
test_set = test_data.values # 읽은 값을 list형태로 변경해줌(.values)
# size : (371(날짜), 6(요소 수))
# [6(2018.12.26~31) + 365(2019년) = 371]
# 6의 의미 : 제작하고자 하는 모델은 '월~금'(ex 2018년 12월 26일~1월 1일)파악해서 그 다음 '월'(ex 2019년 1월 2일) 예측하는 모델
#            2019년 1월 2일부터 12월 31일을 예측하는 모델을 만들기 위해서 2018.12.26~31의 데이터를 추가함.

test_set_np = np.array(test_set) # 계산을 쉽게 하기 위해 numpy를 사용 > numpy array형태로 변환

change_test = np.reshape(test_set_np,(53,7,data_parameter)) #(371x6) >> (53x7(일주일)x6)로 차원변경 (요일별로 정렬해줌)
# 차원변경하는 이유 : 주간/주말을 나누기 위해서
# ex) change_test[0] >> 2018/12/26~2019/1/1 이 들어가 있으며 2018.12.26은 수요일이다. 
'''
change_test에는 데이터가 이런식으로 들어가 있다.
수 목 금 토 일 월 화
26 27 28 29 30 31 1
2  3  4 ...해서 12/31데이터까지
'''

# 1-1. 주말/주간 나눠주기
# testset은 '수 목 금 토 일 월 화' 순서대로 저장되어 있음. 

#(주말나누기)
# [3]번째 : 토요일, [4]번째 : 일요일
weekend_test = change_test[:, 3:5,:] # 토~일 자르기
#weekend_test에는 '토 일'만 들어가 있음.

#(주간나누기)
sat = np.delete(change_test,3,axis=1) # [3]번째(토요일) 삭제
week_test = np.delete(sat,3,axis=1)   # [4]번째(일요일) 삭제
#week_test에는 '수 목 금 월 화'만 남게 됨.

# 주말/주간 나눠준 데이터를 다시 2차원으로 변형
WEEK_test = np.reshape(week_test,(265,data_parameter))       # (53X5,6)
WEEKEND_test = np.reshape(weekend_test,(106,data_parameter)) # (53X2,6)


# 2. 모델제작(class 선언)_CNN(Convolutional Neural Network)+LSTM
#(주간데이터가 거칠 class)
class Power_week(torch.nn.Module): 
    def __init__(self, input_dim, hidden_dim, output_dim, layers): # 사용할 layer들 선언
        # input_dim, hidden_dim, output_dim, layers은 LSTM의 입력을 위해 받는 값.
        super(Power_week, self).__init__()
        
        # Convolution layer 3개 사용 (1>>32>>128>>64)
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
        #fully-connected를 사용해서 size가 5(주간)인 1차원으로 만듦.
        self.fc_cnn = torch.nn.Linear(64, 5, bias=True)
        torch.nn.init.xavier_uniform_(self.fc_cnn.weight) #가중치 초기화(xavier사용)

        # LSTM 사용
        self.LSTM = torch.nn.LSTM(input_dim, hidden_week_dim, num_layers=layers, batch_first=True) 
        #fully-connected를 사용해서 output을 size 1(전력사용량)의 1차원으로 만듦.
        self.fc_rnn = torch.nn.Linear(hidden_week_dim, output_dim, bias=True) #하이퍼 파라미터에서 각각 5(주간)과 1로 선언했음.

    def forward(self, x): # class사용시 수행과정
    
        x = torch.unsqueeze(x,1) # 차원맞추기 위해
        w,h = x.shape[2],x.shape[3] 
        
        # CNN
        out = self.layer1(x) 
        out = self.layer2(out)
        out = self.layer3(out) # layer 3개 거침
        out = out.view(out.size(0), -1) # 차원맞추기 위해
        out = self.fc_cnn(out) # fully-connected로 1차원 변형
        out = torch.unsqueeze(out,-1) # 차원맞추기 위해 

        # LSTM
        x, _status = self.LSTM(out) 
        x = self.fc_rnn(x[:, -1]) # fc를 거쳐 최종적으로 주간의 전력사용량을 예측한 값이 나옴.
        return x

#(주말데이터가 거칠 class)
# 설명은 위의 Power_weekend에서와 같다. 하지만 주말데이터와 주간데이터의 사이즈가 다르기 때문에 
# 연산(kernal_size, stride, Maxpooling여부 등)과 차원(5>>2)에 있어서 약간의 변화만 있기 때문에 생략한다.
class Power_weekend(torch.nn.Module): 
    def __init__(self, input_dim, hidden_dim, output_dim, layers):
        super(Power_weekend, self).__init__() 
        self.layer1 = torch.nn.Sequential(
            torch.nn.Conv2d(1, 32, kernel_size=2, stride=1, padding=1), 
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=1))
        self.layer2 = torch.nn.Sequential(
            torch.nn.Conv2d(32, 128, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer3 = torch.nn.Sequential(
            torch.nn.Conv2d(128, 64, kernel_size=2, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2))
        self.fc_cnn = torch.nn.Linear(64, 2, bias=True)
        torch.nn.init.xavier_uniform_(self.fc_cnn.weight)
        self.LSTM = torch.nn.LSTM(input_dim, hidden_weekend_dim, num_layers=layers, batch_first=True) 
        self.fc_rnn = torch.nn.Linear(hidden_weekend_dim, output_dim, bias=True) 

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
        x = self.fc_rnn(x[:, -1]) # fc를 거쳐 최종적으로 주말의 전력사용량을 예측한 값이 나옴.   
        return x


# 3. 입력형식을 맞추기 위한 정규화 및 데이터구조변경
# trainset과 testset 둘다 통과하는 함수이며 주간, 주말에 따라서만 다르게 적용시킨다

# 데이터구조변경 함수 - 주간
# '월~금'을 보고 '월'예측,'화~(주말뛰고)월'을 보고 '화'예측시키기 위해 input데이터를 5개씩 구성되도록 만들어야 한다.
def build_dataset_week(time_series): # WEEK(1305,6)와 WEEK_test(265,6)가 들어오는 함수.
  seq_length = 5 # 5일씩 cut
  dataX = []
  dataY = []
  for i in range(0, len(time_series)-seq_length): #1300번 반복 
    _x = time_series[i:i + seq_length, :-1] # ex) '월~금' 5개씩 자르고, 전력사용량을 제외한 5개의 요소를 _x라고 하겠다.
    _y = time_series[i+seq_length,-1] # _y는 모델이 예측할 '월'의 실제전력사용량(비교와 학습을 위해 뽑음.)
    
    dataX.append(_x)
    dataY.append(_y)
    
  # dataX : 2018.12.26(~1/1) ~ 2019.12.24(~12/30)
  # dataY : 2019년 1월 2일부터 2019월 12월 31일의 (주말을 제외한)실제값 
  return np.array(dataX), np.array(dataY) #numpy형식으로 return 

# 데이터구조변경 함수 - 주말
# '토 일'을 보고 '(그 다음주)토'예측,'일~(다음주)토'을 보고 '일'예측시키기 위해 input데이터를 2개씩 구성되도록 만들어야 한다.
def build_dataset_weekend(time_series): # WEEKEND(522,6)와 WEEKEND_test(106,6)가 들어오는 함수.
  seq_length = 2 #2일씩 cut
  dataX = []
  dataY = []

  for i in range(0, len(time_series)-seq_length): # 520번 반복
    _x = time_series[i:i + seq_length, :-1] # ex) 주말 2개씩 자르고, 전력사용량을 제외한 5개의 요소를 _x라고 하겠다.
    _y = time_series[i+seq_length, -1] # _y는 모델이 예측할 '토'or'일'의 실제전력사용량(비교와 학습을 위해 뽑음.)  
    dataX.append(_x)
    dataY.append(_y)
  # dataX : 2018.12.29(~30) ~ 2019.12.22(,12/28)
  # dataY : 2019년 1월 5일부터 2019월 12월 29일의 주말실제값   
  return np.array(dataX), np.array(dataY) #numpy형식으로 return 

# 선언했던 함수를 이용해 데이터구조변경
#(trainset)
trainX_week, trainY_week = build_dataset_week(WEEK)
trainX_weekend, trainY_weekend = build_dataset_weekend(WEEKEND)

#(testset)
testX_week, testY_week = build_dataset_week(WEEK_test)
testX_weekend, testY_weekend = build_dataset_weekend(WEEKEND_test)

#(데이터 정규화)
#온도, GDP 등 모든 요소는 다른 단위를 가지고 있다. 따라서 각 요소의 단위를 통일해야 한다. 
#아래는 실제전력사용량에 미치는 정도를 0~1의 숫자로 바꿔주는 과정이다.
train_x_week_numerator = trainX_week - np.min(trainX_week, 0) 
train_x_week_denominator = np.max(trainX_week, 0) - np.min(trainX_week, 0)
train_x_week_set = train_x_week_numerator / (train_x_week_denominator + 1e-7)
train_y_week_numerator = trainY_week - np.min(trainY_week, 0) 
train_y_week_denominator = np.max(trainY_week, 0) - np.min(trainY_week, 0)
train_y_week_set = train_y_week_numerator / (train_y_week_denominator + 1e-7)
#최대~최소값을 전체로 잡고, 각 데이터들의 값을 0~1로 표현한다.

train_x_weekend_numerator = trainX_weekend - np.min(trainX_weekend, 0) 
train_x_weekend_denominator = np.max(trainX_weekend, 0) - np.min(trainX_weekend, 0)
train_x_weekend_set = train_x_weekend_numerator / (train_x_weekend_denominator + 1e-7)
train_y_weekend_numerator = trainY_weekend - np.min(trainY_weekend, 0) 
train_y_weekend_denominator = np.max(trainY_weekend, 0) - np.min(trainY_weekend, 0)
train_y_weekend_set = train_y_weekend_numerator / (train_y_weekend_denominator + 1e-7)

test_x_week_numerator = testX_week - np.min(testX_week, 0) 
test_x_week_denominator = np.max(testX_week, 0) - np.min(testX_week, 0)
test_x_week_set = test_x_week_numerator / (test_x_week_denominator + 1e-7)
test_y_week_numerator = testY_week - np.min(testY_week, 0) 
test_y_week_denominator = np.max(testY_week, 0) - np.min(testY_week, 0)
test_y_week_set = test_y_week_numerator / (test_y_week_denominator + 1e-7)

test_x_weekend_numerator = testX_weekend - np.min(testX_weekend, 0) 
test_x_weekend_denominator = np.max(testX_weekend, 0) - np.min(testX_weekend, 0)
test_x_weekend_set = test_x_weekend_numerator / (test_x_weekend_denominator + 1e-7)
test_y_weekend_numerator = testY_weekend - np.min(testY_weekend, 0) 
test_y_weekend_denominator = np.max(testY_weekend, 0) - np.min(testY_weekend, 0)
test_y_weekend_set = test_y_weekend_numerator / (test_y_weekend_denominator + 1e-7)

# 학습을 시키기 위해서는 tensor형태로 만들어주어야 한다.(tensor 변형)
trainX_week_tensor = torch.FloatTensor(train_x_week_set) 
trainY_week_tensor = torch.FloatTensor(train_y_week_set)
trainX_weekend_tensor = torch.FloatTensor(train_x_weekend_set) 
trainY_weekend_tensor = torch.FloatTensor(train_y_weekend_set)

testX_week_tensor = torch.FloatTensor(test_x_week_set)
testY_week_tensor = torch.FloatTensor(test_y_week_set)
testX_weekend_tensor = torch.FloatTensor(test_x_weekend_set)
testY_weekend_tensor = torch.FloatTensor(test_y_weekend_set)


# 4.선언해주었던 class 실행 및 loss설정 
Power_week_prediction = Power_week(1, hidden_week_dim, output_dim, 1)
Power_weekend_prediction = Power_weekend(1, hidden_weekend_dim, output_dim, 1)

criterion = torch.nn.MSELoss() # MSE loss설정

week_optimizer = optim.Adam(Power_week_prediction.parameters(), lr=learning_rate) # Adam optimizer사용
weekend_optimizer = optim.Adam(Power_weekend_prediction.parameters(), lr=learning_rate)


# 5. Training (2014~2018년 학습하기)
#(trainset-주간학습)
for i in range(100+1): # 1000번 학습하라
    X = trainX_week_tensor # 5개 요소
    Y = trainY_week_tensor # 비교할 실제 전력사용량
    
    week_optimizer.zero_grad()
    outputs = Power_week_prediction(X) #5개 요소로 예측한 예측값(outputs)
    outputs = torch.squeeze(outputs,dim=1)

    loss = criterion(outputs, Y) # 예측값과 실제값을 앞서 지정해준 MSE loss로 비교
    loss.backward() 
    week_optimizer.step() # 다음 step으로 update(learning rate)
    
    if i % 100 == 0 :
      print(i, "loss: ", loss.item()) # 100번 계산할 때마다 오차값 표시

first = outputs[-1] # trainset에 일주일단위를 맞추기 위해서 2019년 1월 1일을 포함시켜줬었음.
# 따라서 train에서 예측한 예측값(outputs)의 마지막에 있는 값은 2019년 1월 1일을 예측한 값.

#(testset-주말학습)_위와동일
for i in range(100+1): #iterations
    X_ = trainX_weekend_tensor
    Y_ = trainY_weekend_tensor
    
    weekend_optimizer.zero_grad()
    outputs_ = Power_weekend_prediction(X_)
    outputs_ = torch.squeeze(outputs_,dim=1)

    loss_ = criterion(outputs_, Y_) 
    loss_.backward() 
    weekend_optimize.step()

    if i % 100 == 0 :
      print(i, "loss: ", loss_.item()) 


# 6. Test (2019년 예측하기)
with torch.no_grad():
    #(주간예측)
    prediction_week = Power_week_prediction(testX_week_tensor) # test주간데이터 넣어서 예측값(prediction_week) 뽑음 
    prediction_week = torch.squeeze(prediction_week,dim=-1)

    #예측값은 정규화과정으로 인해 0~1로 표현된 값임. 이를 되돌려줘야 한다.(실제전력사용량 단위)
    prediction_powerV_week = prediction_week *(test_y_week_denominator + 1e-7) + np.min(testY_week, 0) #전력사용량 단위로 변경
    First = first *(test_y_week_denominator + 1e-7) + np.min(testY_week, 0) #2019년 1월 1일 값도 마찬가지

    #(주말예측)_위와 동일
    prediction_weekend = Power_weekend_prediction(testX_weekend_tensor)
    prediction_weekend = torch.squeeze(prediction_weekend,dim=-1)
    prediction_powerV_weekend = prediction_weekend *(test_y_weekend_denominator + 1e-7) + np.min(testY_weekend, 0) #실제 전력예측량(365)
    
    # MAPE 구하기
    MAPE_weekend_whole =  np.mean(np.abs((testY_weekend - prediction_powerV_weekend.numpy()) / testY_weekend)) * 100
    MAPE_week_whole =  np.mean(np.abs((testY_week - prediction_powerV_week.numpy()) / testY_week)) * 100
    final = (MAPE_week_whole*260 + MAPE_weekend_whole*104)/364 # 주말과 주간 MAPE 합치기


# 7. 출력하기
    print('============평일============')
    print(MAPE_week_whole)
    print('============주말============')
    print(MAPE_weekend_whole)
print('============ 결론 ============')
print(final)
print("1월 1일")
print(First)

# 출력데이터를 엑셀로 저장하기 위한 코드
df3 = pd.DataFrame(prediction_powerV_week)
df3.to_excel('평일_pred.xlsx', index=False)
df6 = pd.DataFrame(prediction_powerV_weekend)
df6.to_excel('주말_pred.xlsx', index=False)
