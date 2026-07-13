from flask import Flask, render_template, request
import numpy as np
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import io
import base64

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("web.html")


@app.route("/calculate", methods=["POST"])
def calculate():

    try:
        x = np.array([float(i) for i in request.form["x"].split()])
        y = np.array([float(i) for i in request.form["y"].split()])

        if len(x) != len(y):
            return "X and Y must contain same number of values."

        X = x.reshape(-1,1)

        model = LinearRegression()
        model.fit(X,y)

        y_pred = model.predict(X)

        slope = model.coef_[0]
        intercept = model.intercept_

        mse = mean_squared_error(y,y_pred)
        mae = mean_absolute_error(y,y_pred)
        r2 = r2_score(y,y_pred)

        # Graph
        plt.figure(figsize=(6,4))
        plt.scatter(x,y,color='blue',label="Actual Data")
        plt.plot(x,y_pred,color='red',label="Regression Line")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.title("Linear Regression")
        plt.legend()

        img = io.BytesIO()
        plt.savefig(img, format='png')
        img.seek(0)

        graph = base64.b64encode(img.getvalue()).decode()

        plt.close()

        return render_template(
            "web.html",
            equation=f"Y = {slope:.4f}X + {intercept:.4f}",
            mse=f"{mse:.4f}",
            mae=f"{mae:.4f}",
            r2=f"{r2:.4f}",
            graph=graph
        )

    except Exception as e:
        return str(e)


if __name__ == "__main__":
    app.run(debug=True)