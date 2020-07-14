import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math

file = pd.read_csv('hour.csv')

dataset = file.copy()

dataset1 = dataset.drop(['index','date','casual','registered'], axis=1)

print(dataset1.isnull().sum())

dataset1.hist(rwidth =0.9)
plt.tight_layout()

plt.subplot(2,2,1)
plt.title('Temp vs demand')
plt.scatter(dataset1['temp'], dataset1['demand'] , s=2, c='g')


plt.subplot(2,2,2)
plt.title('aTemp vs demand')
plt.scatter(dataset1['atemp'], dataset1['demand'] , s=2, c='b')


plt.subplot(2,2,3)
plt.title('humidity vs demand')
plt.scatter(dataset1['humidity'], dataset1['demand'] , s=2, c='r')



plt.subplot(2,2,4)
plt.title('Windspeed vs demand')
plt.scatter(dataset1['windspeed'], dataset1['demand'] , s=2, c='m')


plt.tight_layout()

c = ['r','g','b','m']

plt.subplot(3,3,1)
plt.title('Avg demand per season')
x = dataset1['season'].unique()
y = dataset1.groupby('season').mean()['demand']
plt.bar(x,y,color = c)




plt.subplot(3,3,2)
plt.title('Avg demand per year')
x = dataset1['year'].unique()
y = dataset1.groupby('year').mean()['demand']
plt.bar(x,y,color = c)




plt.subplot(3,3,3)
plt.title('Avg demand per month')
x = dataset1['month'].unique()
y = dataset1.groupby('month').mean()['demand']
plt.bar(x,y,color = c)



plt.subplot(3,3,4)
plt.title('Avg demand per hour')
x = dataset1['hour'].unique()
y = dataset1.groupby('hour').mean()['demand']
plt.bar(x,y,color = c)



plt.subplot(3,3,5)
plt.title('Avg demand per holiday')
x = dataset1['holiday'].unique()
y = dataset1.groupby('holiday').mean()['demand']
plt.bar(x,y,color = c)



plt.subplot(3,3,6)
plt.title('Avg demand per weekday')
x = dataset1['weekday'].unique()
y = dataset1.groupby('weekday').mean()['demand']
plt.bar(x,y,color = c)



plt.subplot(3,3,7)
plt.title('Avg demand per workingday')
x = dataset1['workingday'].unique()
y = dataset1.groupby('workingday').mean()['demand']
plt.bar(x,y,color = c)



plt.subplot(3,3,8)
plt.title('Avg demand per weather')
x = dataset1['weather'].unique()
y = dataset1.groupby('weather').mean()['demand']
plt.bar(x,y,color = c)


plt.tight_layout()




print(dataset1['demand'].describe())
print(dataset1['demand'].quantile([0.05, 0.1, 0.15, 0.9, 0.95, 0.99]))


newdata = dataset1.copy()

corr = newdata[['temp','humidity', 'atemp', 'windspeed', 'demand']].corr()


newdata1=newdata.drop(['windspeed', 'atemp' ,'workingday','year', 'weekday'], axis=1)


acorr = pd.to_numeric(newdata1['demand'], downcast ='float')
plt.figure()
plt.acorr(acorr, maxlags = 12)

df1 = newdata1['demand']
df2 = np.log(df1)

plt.figure()
plt.hist(df1, rwidth=0.9, bins=20)
plt.figure()
plt.hist(df2, rwidth =0.9, bins = 20)


newdata1['demand'] =np.log(newdata['demand'])

t1 = newdata1['demand'].shift(+1).to_frame()
t1.columns = ['t-1']


t2 = newdata1['demand'].shift(+2).to_frame()
t2.columns = ['t-2']



t3 = newdata1['demand'].shift(+3).to_frame()
t3.columns = ['t-3']


newdatalag = pd.concat([newdata1, t1, t2, t3], axis=1)

newdatalag = newdatalag.dropna()


print(newdatalag.dtypes)

newdatalag['season']= newdatalag['season'].astype('category')
newdatalag['month']= newdatalag['month'].astype('category')
newdatalag['hour']= newdatalag['hour'].astype('category')
newdatalag['holiday']= newdatalag['holiday'].astype('category')
newdatalag['weather']= newdatalag['weather'].astype('category')


dummy = pd.get_dummies(newdatalag, drop_first = True) 



X = dummy.drop(['demand'], axis =1)
Y = dummy[['demand']]

size = 0.7 * len(X)
size1 = int(size)

x_train = X.values[0 : size1]
x_test = X.values[size1 : len(X)]


y_train = Y.values[0 : size1]
y_test = Y.values[size1 : len(Y)]

from sklearn.linear_model import LinearRegression

reg = LinearRegression()

reg.fit(x_train, y_train)



r2_train = reg.score(x_train, y_train)
r2_test = reg.score(x_test, y_test)


y_predict = reg.predict(x_test)

from sklearn.metrics import mean_squared_error
rmse = math.sqrt(mean_squared_error(y_test, y_predict))


y_teste = []
y_predicte = []

for i in range(0, len(y_test)):
    y_teste.append(math.exp(y_test[i]))
    y_predicte.append(math.exp(y_predict[i]))

print(len(y_teste))
print(len(y_test))

sum = 0.0
for i in range(0, len(y_teste)):
    loga = math.log(y_teste[i]+1) 
    logp = math.log(y_predicte[i]+1)
    diff = (logp - loga)**2
    sum = sum + diff
    
rmsle1 = math.sqrt(sum/len(y_test))

print(rmsle1)    
    