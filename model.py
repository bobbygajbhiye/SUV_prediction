import pandas as pd

data=pd.read_csv(r'C:\Users\DELL\Desktop\dev\models\SUV_Bobby\SUV.txt')

data.drop('User ID',axis=1,inplace=True)

data['Gender']=pd.get_dummies(data['Gender'],drop_first=True)

X=data.drop('Purchased',axis=1).values
y=data['Purchased']

from sklearn.preprocessing import StandardScaler
X=StandardScaler().fit_transform(X)

from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier(metric='euclidean', n_neighbors=10)
knn.fit(X,y)

import pickle
pickle.dump(knn,open(r'C:\Users\DELL\Desktop\dev\models\SUV_Bobby\SUV_model.pkl','wb'))
model=pickle.load(open(r'C:\Users\DELL\Desktop\dev\models\SUV_Bobby\SUV_model.pkl','rb'))