from flask import Flask,render_template,request
import pickle
import json
import numpy as np

with open (r'C:\Users\Admin\Desktop\python projects\project_dave\final_rf.pkl','rb')as file:
    model=pickle.load(file)

app=Flask(__name__)

@app.route('/')
def index():
    return render_template ('index.html')

@app.route('/predict',methods=["GET","POST"])
def predict():
    height=request.form['height']
    age=request.form['age']
    weigth=request.form['weight']

    user_data=np.zeros(3)

    user_data[0]=height
    user_data[1]=age
    user_data[2]=weigth

    print(user_data)
    result=model.predict([user_data])
    print(result)
    
    return render_template ('index.html',prediction=result)

if __name__=='__main__':
    app.run(host='0.0.0.0',port=8080,debug=True)
