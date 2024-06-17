from flask import Flask, render_template, request, redirect, url_for

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    # Perform prediction or any other processing here
    # Redirect to the appropriate location

    # Your prediction code here

    return render_template('prediction.html', y_test=y_test, y_predicted=y_predicted)

if __name__ == '__main__':
    app.run(debug=True)
    