import numpy as np
from wsgiref import simple_server
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS, cross_origin
import pickle


@cross_origin()
app = Flask(__name__)
model = pickle.load(open('model_01.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index1.html')

@app.route('/predict',methods=['POST','Get'])
def predict():

    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)
    print(prediction)
    output = prediction[0]

    return render_template('index1.html', prediction_text='Cardio - {}'.format(output))

@app.route('/results',methods=['POST'])
def results():

    data = request.get_json(force=True)
    prediction = model.predict([np.array(list(data.values()))])

    output = prediction[0]
    return jsonify(output)

port = int(os.getenv("PORT"))
if __name__ == "__main__":
    host = '0.0.0.0'
    #port = 5000
    httpd = simple_server.make_server(host, port, app)
    print("Serving on %s %d" % (host, port))
    httpd.serve_forever()