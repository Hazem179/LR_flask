import numpy as np
from flask import Flask,render_template,request,jsonify
import pickle

app = Flask(__name__)
model = pickle.load(open('model.pkl','rb'))


@app.route('/')
def home():
    return render_template("index.html")
@app.route('/predict',methods=["POST"])
def predict():
    features=[int(x) for x in request.form.values()]
    final_features = [np.array(features)]
    prediction = model.predict(final_features)
    result = round(prediction[0],2)
    print(result)
    return render_template('index.html',prediction_text = f'House Price ${result}')


if __name__ == '__main__':
    app.run()
