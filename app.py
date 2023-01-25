from flask import Flask, request, render_template, url_for
import pickle
import pandas as pd
import numpy as np
import joblib
scaler = joblib.load("scaler.save")


app = Flask(__name__)
model = pickle.load(open('modelfixRF.pkl', 'rb'))


@app.route("/home")
@app.route("/")
def hello():
    return render_template("index.html")


@app.route("/predict", methods=["GET", "POST"])
def predict():
    if request.method == "POST":
        input_features = [float(x) for x in request.form.values()]
        features_value = [np.array(input_features)]
        feature_names = ["ph", "Hardness", "Solids", "Chloramines", "Sulfate",
                         "Conductivity", "Organic_carbon", "Trihalomethanes", "Turbidity"]

        df = pd.DataFrame(features_value, columns=feature_names)
        df = scaler.transform(df)
        output = model.predict(df)

        if output[0] == 1:
            prediction = "Aman"
        else:
            prediction = "Tidak Aman"

        return render_template('output.html', prediction_text="Air {} Untuk Dikonsumsi ".format(prediction), ph=features_value[0][0], Hardness=features_value[0][1],Solids=features_value[0][2], Chloramines=features_value[0][3], Sulfate=features_value[0][4], Conductivity=features_value[0][5], Organic_carbon=features_value[0][6], Trihalomethanes=features_value[0][7], Turbidity=features_value[0][8])


if __name__ == "__main__":
    app.run(debug=True)
