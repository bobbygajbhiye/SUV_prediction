from flask import Flask, render_template, request
from sklearn.preprocessing import StandardScaler
import pickle

model=pickle.load(open('SUV_model.pkl','rb'))

app = Flask(__name__)
@app.route('/output',methods=['POST'])
def post_fun():
    gender=int(request.form['gen'])
    age=int(request.form['a'])
    salary=int(request.form['sal'])
    pred=model.predict(StandardScaler().fit_transform([[gender,age,salary]]))
    return render_template('output.html',prediction=pred)

@app.route('/')
def home():
	return render_template('input.html')

if __name__=='__main__':
	app.run()
