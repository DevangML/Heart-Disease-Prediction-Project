from importlib.resources import path
from flask import Flask, render_template, request
import numpy as np
import pickle
from pathlib import Path
import joblib
import os


def relpath(p): return os.path.normpath(
    os.path.join(os.path.dirname(__file__), p))


fPath = relpath('./heart_model.pkl')

webapp = Flask(__name__)
model = pickle.load(open(fPath, 'rb'))


@webapp.route("/",)
def hello():
    return render_template("index.html")


@webapp.route("/detail", methods=["POST"])
def submit():
    if request.method == "POST":
        name = request.form["Username"]

    return render_template("detail.html", n=name)


@webapp.route("/predict", methods=["POST"])
def predict():
    if request.method == "POST":

        age = int(request.form['age'])
        sex = int(request.form['sex'])
        cp = int(request.form['cp'])
        trestbps = int(request.form['trestbps'])
        chol = int(request.form['chol'])
        fbs = int(request.form['fbs'])
        restecg = int(request.form['restecg'])
        thalach = int(request.form['thalach'])
        exang = int(request.form['exang'])
        oldpeak = int(request.form['oldpeak'])
        slope = int(request.form['slope'])
        ca = int(request.form['ca'])
        thal = int(request.form['thal'])

        values = np.array([[age, sex, cp, trestbps, chol, fbs,
                          restecg, thalach, exang, oldpeak, slope, ca, thal]])
        prediction = model.predict(values)

        return render_template('predict.html', prediction=prediction)


if __name__ == "__main__":
    webapp.run()
