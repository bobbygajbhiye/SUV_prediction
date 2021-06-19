import pandas as pd

data=pd.read_csv(r'C:\Users\DELL\Desktop\dev\models\SUV_Bobby\SUV.txt')

data.drop('User ID',axis=1,inplace=True)

data['Gender']=pd.get_dummies(data['Gender'],drop_first=True)

X=data.drop('Purchased',axis=1)
y=data['Purchased']

from sklearn.tree import DecisionTreeClassifier
dtc=DecisionTreeClassifier(criterion='gini',max_depth=30,max_features=1,max_leaf_nodes=7,min_samples_split=5)
dtc.fit(X,y)

import pickle
pickle.dump(dtc,open(r'C:\Users\DELL\Desktop\dev\models\SUV_Bobby\SUV_model.pkl','wb'))
model=pickle.load(open(r'C:\Users\DELL\Desktop\dev\models\SUV_Bobby\SUV_model.pkl','rb'))