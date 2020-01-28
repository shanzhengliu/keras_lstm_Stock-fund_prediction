from keras.models import load_model
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
from keras.layers import Dense,LSTM,Dropout
from numpy import array
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
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
model = load_model('my_model.h5')
predictY = model.predict(features_set)
print(features_set.shape)
last = features_set[len(features_set)-1].reshape(1, 15, 1)
print(last.shape)
list = []
i=0
while(i<60):
    templabel= model.predict(last)
    list.append(float(templabel))
    last = last.reshape(15)
    last = np.delete(last,0)
    

    last=np.append(last,templabel)
    
    last = last.reshape(1,15,1)
    i=i+1
    
   
list= np.array(list)
list.dtype="float"

list=np.reshape(list,(len(list),1))

predictY=np.vstack((predictY,list))

plt.figure(figsize=(10,5))
plt.plot(labels)
plt.plot(predictY)
plt.legend()
plt.show()
