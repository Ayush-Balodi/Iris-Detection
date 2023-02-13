import os
import pickle 
from flask import Flask
from flask_restful import Resources, Api, reqparse
from flask_cors import CORS
import numpy as np 

app = Flask(__name__)
CORS(app)
api = Api(app)

parser = reqparse.RequestParser()
parser.add_argument("sepal_length")
parser.add_argument("sepal_width")
parser.add_argument("petal_length")
parser.add_argument("petal_width")

if os.path.isfile("./ourmodel.pkl"):
    model = pickle.load(open("./ourmodel.pkl", "rb"))
else:
    raise FileNotFoundError

class Predict(Resource):
    def post(self):
        args = parser.parse_args()

        X = (np.array([args["sepal_length"],
                    args["sepal_width"],
                    args["petal_length"],
                    args["petal_width"]
                    ]
                    ).astype("float").reshape(1,-1)
            )
        _y = model.predict(X)[0]
        return {"class": _y}
api.add_resource(Predict, "/predict")
if __name__ == "__main__":
    app.run(debug = True)