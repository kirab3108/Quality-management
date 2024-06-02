#!/usr/bin/env python
# coding: utf-8

# # Приложение №1 к Курсовой работе. Булдакова К.А., ЗБ-ПИ20-2.

# 1. Установка необходимых инструментов

# In[ ]:


#pip install yfinance
#pip install pandas_datareader
#pip install keras
#pip install tensorflow


# 2. Импорт инструментов

# In[2]:


import math
import pandas as pd
import pandas_datareader as web
import numpy as np

from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
import yfinance as yf


# 3. Импорт данных

# In[ ]:


df = yf.download('AAPL', start='2010-01-01', end='2020-12-26')


# In[4]:


df


# In[5]:


df.shape


# 4. Визуализация полученных данных

# In[7]:


plt.figure(figsize=(16,8))
plt.title('Полученные данные о цене закрытия акций')
plt.plot(df['Close'])
plt.xlabel('Год',fontsize=18)
plt.ylabel('Цена закрытия ($)',fontsize=18)
plt.show()


# 5. Подготовка выборки данных

# In[8]:


#Создаем массив с информацией только из столбца "Close"
data = df.filter(['Close'])
dataset = data.values
#Устанавливаем количество строк
training_data_len = math.ceil( len(dataset) *.8)


# In[23]:


#Масштабируем данные, то есть оставляем цифры в диапазоне от 0 до 1
scaler = MinMaxScaler(feature_range=(0, 1)) 
scaled_data = scaler.fit_transform(dataset)


# In[24]:


train_data = scaled_data[0:training_data_len  , : ]
#Задаем условия для x_train and y_train
x_train=[]
y_train = []
for i in range(100,len(train_data)):
    x_train.append(train_data[i-100:i,0])
    y_train.append(train_data[i,0])


# In[25]:


x_train, y_train = np.array(x_train), np.array(y_train)


# In[26]:


#Приводим к LSTM
x_train = np.reshape(x_train, (x_train.shape[0],x_train.shape[1],1))


# 5. Создание и обучение модели

# In[32]:


#Модель LSTM: 2 слоя с 50 нейронами и два слоя Dense - один с 25 нейронами и с 1 нейроном. 
model = Sequential()
model.add(LSTM(units=50, return_sequences=True,input_shape=(x_train.shape[1],1)))
model.add(LSTM(units=50, return_sequences=False))
model.add(Dense(units=25))
model.add(Dense(units=1))


# In[33]:


#Компилируем модель
model.compile(optimizer='adam', loss='mean_squared_error')


# In[34]:


#Обучаем модель
model.fit(x_train, y_train, batch_size=1, epochs=1)


# 6. Тестирование

# In[42]:


#Создаем данные test_data
test_data = scaled_data[training_data_len - 100: , : ]
x_test = []
y_test =  dataset[training_data_len : , : ] 
for i in range(100,len(test_data)):
    x_test.append(test_data[i-100:i,0])


# In[43]:


x_test = np.array(x_test)


# In[44]:


x_test = np.reshape(x_test, (x_test.shape[0],x_test.shape[1],1))


# 7. Получение прогнозируемых данных

# In[45]:


predictions = model.predict(x_test) 
predictions = scaler.inverse_transform(predictions)


# In[46]:


#Среднеквадратичная ошибка
rmse=np.sqrt(np.mean(((predictions- y_test)**2)))
rmse


# 8. Визуализация спрогнозированных данных

# In[47]:


train = data[:training_data_len]
valid = data[training_data_len:]
valid['Predictions'] = predictions
#Visualize the data
plt.figure(figsize=(16,8))
plt.title('Визуализация работы модели')
plt.xlabel('Год', fontsize=18)
plt.ylabel('Цена закрытия ($)', fontsize=18)
plt.plot(train['Close'])
plt.plot(valid[['Close', 'Predictions']])
plt.legend(['Обучение', 'Значения', 'Прогноз'], loc='lower right')
plt.show()


# In[49]:


#вывод - сравнение данных
valid

