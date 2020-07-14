import pandas as pd
import os
import pickle
from sklearn.tree import DecisionTreeClassifier

iris=pd.read_csv("C:/Users/snair/Documents/Data Science Assignment/Data Sets/iris.csv")

X=iris.iloc[:,:-1]
Y=iris.iloc[:,-1]

model=DecisionTreeClassifier(criterion="entropy")
model.fit(X,Y)

os.chdir('C:\\Users\\snair\\Documents\\Data Science Assignment\\Practice\\Python\\Deployment\\heroku_demo')

pickle.dump(model,open('model.pkl','wb'))

model=pickle.load(open('model.pkl','rb'))
pred=model.predict([[2.5,3.5,4.1,1.3]])
print(pred[0])
