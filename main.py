
import requests
import time
import execjs

def getUrl(fscode):
  head = 'http://fund.eastmoney.com/pingzhongdata/'
  tail = '.js?v='+ time.strftime("%Y%m%d%H%M%S",time.localtime())
  
  return str(head+fscode+tail)

def getWorth(fscode):
    #用requests获取到对应的文件
    content = requests.get(getUrl(fscode))
    
    print(getUrl(fscode))
   #使用execjs获取到相应的数据
    jsContent = execjs.compile(content.text)
   
    name = jsContent.eval('fS_name')
   
    code = jsContent.eval('fS_code')
    #单位净值走势
    netWorthTrend = jsContent.eval('Data_netWorthTrend')
    #累计净值走势
    ACWorthTrend = jsContent.eval('Data_ACWorthTrend')

    netWorth = []
    ACWorth = []

   #提取出里面的净值
    for dayWorth in netWorthTrend[::-1]:
        netWorth.append(dayWorth['y'])

    for dayACWorth in ACWorthTrend[::-1]:
        ACWorth.append(dayACWorth[1])
    print(name,code)
    return netWorth, ACWorth


netWorth, ACWorth = getWorth("005224")

mydata = netWorth[::-1];



import keras
from keras.models import Sequential
from keras.layers import Dense,LSTM,Dropout,GRU
from numpy import array
import tensorflow as tf
import numpy as np
from keras.optimizers import Optimizer,SGD,Adam,Adadelta
# data = array(mydata)
# data = data.reshape(1,len(mydata),1)
data = mydata
i = 0;
x = [];
y = [];
while i < len(data)-16:
    x.append(data[i:i+15])
    y.append(data[i+16])
    i=i+1;


features_set, labels = np.array(x), np.array(y)
features_set = np.reshape(features_set, (features_set.shape[0], features_set.shape[1], 1))
# lables = np.reshape(labels,(labels.shape[0],1))
# print(features_set.shape)

model = Sequential()
model.add(LSTM(128,activation='relu',input_shape=(features_set.shape[1], 1),return_sequences = True))
# model.add(Dropout(0.2))

model.add(LSTM(64, return_sequences=True,activation='relu'))
# model.add(Dropout(0.2))
model.add(GRU(32,activation='relu'))

model.add(Dense(1))
optimizer=Adam(lr=0.005)

model.compile(optimizer=optimizer, loss='mean_squared_error')

history = model.fit(features_set, labels, nb_epoch=100, batch_size=30)
score = model.evaluate(features_set, labels, batch_size=128)


model.save('my_model.h5')
import matplotlib.pyplot as plt
# plt.figure(figsize=(10,5))
# plt.plot(netWorth[:7][::-1])
# plt.show()
print(history.history.keys())
print(score)
epochs = len(history.history['loss'])
plt.plot(range(epochs), history.history['loss'], label='loss')

plt.legend()
plt.show()

# last = features_set[len(features_set)-1]
predictY = model.predict(features_set)
# # for  i in range(0,20,1):
# #     templabel= model.predict(last)
# #     print(templabel)
    


plt.figure(figsize=(10,5))
plt.plot(labels)
plt.plot(predictY)
plt.legend()
plt.show()




# def generator_myJijin(currentlist,daymore)
