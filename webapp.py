from importlib.resources import path
from flask import Flask, render_template, request
import numpy as np
import pickle
from pathlib import Path
import joblib
import os


def relpath(p): return os.path.normpath(
    os.path.join(os.path.dirname(__file__), p))


fPath = relpath('../heart_model.pkl')

webapp = Flask(__name__)

model = pickle.load(open(fPath, 'rb'))


@app.route("/")
def hello():
    return render_template("index.html")


@app.route("/detail", methods=["POST"])
def submit():
    if request.method == "POST":
        name = request.form["Username"]

    return render_template("detail.html", n=name)


@app.route("/predict", methods=["POST"])
def predict():
    if request.method == "POST":
        age = int(request.form[''])
        sex = int(request.form[''])
        cp = int(request.form[''])
        trestbps = int(request.form[''])
        chol = int(request.form[''])
        fbs = int(request.form[''])
        restecg = int(request.form[''])
        thalach = int(request.form[''])
        exang = int(request.form[''])
        oldpeak = int(request.form[''])
        slope = int(request.form[''])
        ca = int(request.form[''])
        thal = int(request.form[''])

        values = np.array([[age, sex, cp, trestbps, chol, fbs,
                          restecg, thalach, exang, oldpeak, slope, ca, thal]])
        prediction = model.predict(values)

        return render_template('predict.html', prediction=prediction)


if __name__ = "__main__":
    app.run(debug=True)
