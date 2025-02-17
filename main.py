from flask import Flask, request, render_template
import pickle
import numpy as np
import os

app = Flask(__name__, template_folder=os.path.join(os.getcwd(), 'templates'))


# Load the model
with open("iris.pkl", 'rb') as f:
    model = pickle.load(f)


@app.route("/")
def show():
    return render_template('index.html')


@app.route('/predict', methods=['POST', 'GET'])
def predict():
    print(request.form)

    # Extract feature values from the form
    features = [float(x) for x in request.form.values()]

    # Convert features to a NumPy array
    final_features = np.array(features).reshape(1, -1)

    print(final_features)

    # Use the loaded model for prediction
    result = model.predict(final_features)

    # Map the numerical result to flower names
    if result[0] == 0:
        flower = 'SETOSA'
    elif result[0] == 1:
        flower = "VERSICOLOR"
    else:
        flower = "VIRGINICA"

    return render_template('index.html', predicted_flower="Flower is " + flower)


if __name__ == "__main__":
    app.run(debug=True)
