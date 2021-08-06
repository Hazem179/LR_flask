import pickle
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error,r2_score

df = pd.read_csv("kc_house_data.csv")
x=df.iloc[:,3:7]
y=df.iloc[:,2]
x_train , x_test , y_train , y_test = train_test_split(x,y,test_size=0.2,random_state=32)
LR= LinearRegression()
LR.fit(x_train,y_train)
y_predition=LR.predict(x_test)
r2=r2_score(y_test,y_predition)
pickle.dump(LR,open('model.pkl','wb'))
model = pickle.load(open('model.pkl','rb'))
print(x)