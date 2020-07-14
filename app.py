from flask import Flask,render_template,request
import pickle
import numpy as np

app=Flask(__name__)
model=pickle.load(open('model.pkl','rb'))

@app.route('/')
def home():
    return render_template("predict.html")

@app.route('/predict',methods=['POST'])
def predict():
    data=[x for x in request.form.values()]
    features=[np.array(data)]
    pred=model.predict(features)
    
    return render_template("predict.html",predicted="It is a {}".format(pred[0]))

if __name__=="__main__":
    app.run(debug=True)
    

