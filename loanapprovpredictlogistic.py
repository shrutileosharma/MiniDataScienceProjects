import pandas as pd


file = pd.read_csv('01Exercise1.csv')
data = file.copy()

print(data.isnull().sum(axis=0))

cleardata = data.dropna()

cleardata = cleardata.drop(['gender'], axis=1)

print(cleardata.dtypes)

cleardata['ch'] = cleardata['ch'].astype('category')
cleardata['married'] = cleardata['married'].astype('category')
cleardata['status'] = cleardata['status'].astype('category')
print(cleardata.dtypes)


cleardata= pd.get_dummies(cleardata, drop_first= 'True')

from sklearn.preprocessing import StandardScaler

scale = StandardScaler()

cleardata['income'] = scale.fit_transform(cleardata[['income']])
cleardata['loanamt'] = scale.fit_transform(cleardata[['loanamt']])
 
X= cleardata.drop(['status_Y'], axis=1)
Y = cleardata['status_Y']


from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test =   \
train_test_split(X, Y, test_size = 0.3, random_state=1234, stratify = Y )


from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()

lr.fit(x_train, y_train)

y_predict= lr.predict(x_test)


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_predict)

score =lr.score(x_test, y_test)

