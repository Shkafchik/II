from pandas import read_csv
from pandas import DataFrame
import numpy as np
import matplotlib
from matplotlib import pyplot
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense


covid = read_csv(r'C:\Users\Maxim\Desktop\s\Obych\lol1.csv', sep=';', header=0, parse_dates=[0])
covid.iloc[:,1].plot()
print("На данном графике вы увидите, какие значение поступили изначально,т.е с чем предстоить работать программе")
x1 = 320
y2 = 700
plt.legend(fontsize = 12)
plt.plot(x1,y2)
plt.xlabel("День наблюдения", fontsize=14, fontweight="bold")
plt.ylabel("Количество смертей", fontsize=14, fontweight="bold")
pyplot.show()


covid_2 = DataFrame()#Заготовка для таблицы
for i in range(12,0,-1):
        covid_2['t-'+str(i)] = covid.iloc[:,1].shift(i)
  
covid_2['t'] = covid.iloc[:,1].values#записываем исходный ряд в последний столбец

covid_4 = covid_2[40:]#вырезаем из таблицы первые 40 строк, т.к там одни 0


#Отделяем x от y
y = covid_4['t']
X = covid_4.drop('t', axis=1)

#Разделяем на обучающую и тестовую выборку, где на обучающей нейронка учится,
#а на тестовой тестирует свои прогнозы
X_train = X[:272]
y_train = y[:272]
X_test = X[272:]
y_test = y[272:]


#Преобразуем из массивов pandas в numpy
#для работы с keras
X_train = X_train.values
X_test = X_test.values
y_train = y_train.values
y_test = y_test.values

#Создаем модель, по которой будет учиться наша нейронка
model = Sequential()
model.add(Dense(8, input_dim=12, activation='relu'))
model.add(Dense(1, activation='linear'))#1 скрытый слой с линейной выходной функцией

model.compile(loss='mean_squared_error', optimizer='adam',metrics=['mean_absolute_percentage_error'])

model.fit(X_train, y_train, epochs=300, batch_size=32)

#Оценка качества модели на тестовом множестве
scores = model.evaluate(X_test, y_test)
print("\nMape: %.2f%%" %(scores[i]))

#Вычисляем прогноз
predictions = model.predict(X_test)


#Вычисляем подгонку
predictions_train = model.predict(X_train)


x2 = np.arange(0, 272, 1)
x3 = np.arange(272, 286, 1)



plt.plot(x2, y_train, color='blue',label = 'Входные данные')
plt.plot(x2, predictions_train, color='green',label = 'обучение нейронной сети')
plt.plot(x3, y_test, color='blue')
plt.plot(x3, predictions, color='red',label = 'прогноз нейронной сети')
plt.legend(fontsize = 12)
plt.plot(x1,y2)
plt.xlabel("День наблюдения", fontsize=14, fontweight="bold")
plt.ylabel("Количество смертей", fontsize=14, fontweight="bold")
pyplot.show()

