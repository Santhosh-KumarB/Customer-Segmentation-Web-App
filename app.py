from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load the trained K-Means model
model = joblib.load("customer_segmentation_model.pkl")

@app.route("/", methods=["GET", "POST"])
def index():
    cluster = None
    description = None

    if request.method == "POST":
        income = float(request.form["income"])
        spending = float(request.form["spending"])

        features = np.array([[income, spending]])
        cluster = model.predict(features)[0]

        descriptions = {
            0: "Low income – Low spending customer",
            1: "Low income – High spending customer",
            2: "Average income – Average spending customer",
            3: "High income – High spending customer",
            4: "High income – Low spending customer"
        }

        description = descriptions.get(cluster, "Customer segment identified")

    return render_template("index.html", cluster=cluster, description=description)

if __name__ == "__main__":
    app.run(debug=True)
