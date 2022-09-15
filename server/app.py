from ibm_watson import VisualRecognitionV3
from ibm_cloud_sdk_core.authenticators import IAMAuthenticator
from flask import Flask, request, jsonify
from flask_sqlalchemy import SQLAlchemy
from flask_marshmallow import Marshmallow
import os
from tensorflow.python.keras.models import load_model
import cv2
import numpy as np
import json
import bcrypt
from flask_cors import CORS

APP_ROOT = os.path.abspath(os.path.dirname(__file__))

model = load_model('planrsafe_v89.h5')

# Load category names
with open('./diseases.json', 'r') as f:
    category_names = json.load(f)
    img_classes = list(category_names.values())


# Pre-processing images
def config_image_file(_image):
    img = cv2.imread(_image)
    img = cv2.resize(img, (224, 224))
    img = img / 255
    return img


# Predicting
def predict_image(image):
    probabilities = model.predict(np.asarray([image]))[0]
    class_idx = np.argmax(probabilities)
    return {img_classes[class_idx]: probabilities[class_idx],
            'index': class_idx}


# Working as the toString method
def output_prediction(filename):
    _image = f"./leaves/{filename}"

    print(_image)
    img_file = config_image_file(_image)
    prediction = predict_image(img_file)

    disease = list(prediction.keys())[0]
    confidence = list(prediction.values())[0]
    index = list(prediction.values())[1]

    result = {
        "id": str(index),
        "disease": disease,
        "confidence": str(confidence)
    }
    return result


def validate_image(img):
    iam = IAMAuthenticator("lPeaymHBGfbYPAdrfK26nUD9Sv0K2beE5L6BIPQEE62y")
    vr = VisualRecognitionV3(
        version="2018-03-19",
        authenticator=iam
    )
    vr.set_service_url(
        "https://api.us-south.visual-recognition.watson.cloud.ibm.com/instances/e13104ab-293b-490b-a0ea-58280c48ef11")

    file = open(f'./images/{img.filename}', 'rb')
    result = vr.classify(images_file=file).get_result()

    prediction_classes = result["images"][0]["classifiers"][0]["classes"]

    predictions = []
    keys = ['plant', 'tree', 'leaf', 'herb']

    for item in range(len(prediction_classes)):
        predictions.append(result["images"][0]["classifiers"]
                           [0]["classes"][item]["class"])

    matches = [a for a in predictions if a in keys]

    if len(matches) > 0:
        return jsonify({'status': 1, 'predictions': prediction_classes})
    else:
        return jsonify({'status': -1, 'predictions': prediction_classes})


# Init app
app = Flask(__name__)
CORS(app)

# Database
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql://root:password@localhost/plantsafe-db'

# Init db
db = SQLAlchemy(app)
# Init ma
ma = Marshmallow(app)


# Model class User
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(100), unique=True)
    fullname = db.Column(db.String(100))
    password = db.Column(db.String(100))

    def __init__(self, email, fullname, password):
        self.email = email
        self.fullname = fullname
        self.password = password


# User Schema
class UserSchema(ma.Schema):
    class Meta:
        fields = ('id', 'email', 'fullname', 'password')


# Init schema
user_schema = UserSchema()
users_schema = UserSchema(many=True)


# Create a user
@app.route('/api/users', methods=['POST'])
def add_user():
    email = request.json['email']
    fullname = request.json['fullname']
    password = request.json['password'].encode('utf-8')
    hash_password = bcrypt.hashpw(password, bcrypt.gensalt())

    new_user = User(email, fullname, hash_password)
    db.session.add(new_user)
    db.session.commit()

    return user_schema.jsonify(new_user)


# Login user
@app.route('/api/users/login', methods=['POST'])
def login_user():
    email = request.json['email']
    password = request.json['password'].encode('utf-8')

    user = db.session.query(User).filter_by(email=email)
    _user = users_schema.dump(user)

    if len(_user) > 0:
        hashed_password = _user[0]['password'].encode('utf-8')
        if bcrypt.checkpw(password, hashed_password):
            return users_schema.jsonify(user)

    return jsonify({"message": "Invalid credentials"})


# Get All users
@app.route('/api/users', methods=['GET'])
def get_users():
    all_users = User.query.all()
    result = users_schema.dump(all_users)

    return jsonify(result)


# Image file validation
@app.route('/api/validate', methods=['POST'])
def get_prediction_img():
    target = os.path.join(APP_ROOT, 'images/')

    if not os.path.isdir(target):
        os.mkdir(target)

    file = request.files.get('file')

    filename = file.filename
    destination = '/'.join([target, filename])
    print(destination)
    file.save(destination)

    return validate_image(file)


@app.route('/api/predict', methods=['POST'])
def get_something():
    target = os.path.join(APP_ROOT, 'leaves/')

    if not os.path.isdir(target):
        os.mkdir(target)

    file = request.files.get('file')

    filename = file.filename
    destination = '/'.join([target, filename])
    print(destination)
    file.save(destination)

    result = output_prediction(filename)

    return jsonify(result)


# Run Server
if __name__ == '__main__':
    app.run(host="192.168.1.4", port=5000)
