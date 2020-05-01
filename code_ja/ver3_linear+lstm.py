class Power(torch.nn.Module):
  def __init__(self, input_dim, hidden_dim, output_dim, layers):
    super(Power, self).__init__()
    self.linear1 = torch.nn.Linear(4, 256, bias=True)
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

# print(experiment)
plt.plot(testY_tensor)
plt.plot(prediction.data.numpy())
plt.legend(['original', 'prediction'])
plt.show()
