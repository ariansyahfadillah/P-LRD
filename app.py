import tensorflow as tf
import numpy as np
from flask import Flask, render_template, request
from keras.utils import load_img
from keras.utils import img_to_array
from tensorflow import keras

app = Flask(__name__)
model = tf.keras.models.load_model('model.h5')
model.make_predict_function()

@app.route('/', methods=['GET'])
def main():
    return render_template('index.html')

@app.route('/', methods=['POST'])
def predict():
    imagefile = request.files['imagefile']
    imagepath = './static/' + imagefile.filename
    imagefile.save(imagepath)

    image = load_img(imagepath, target_size=(150, 150))
    image = img_to_array(image)

    x = image/255.0
    x = np.array(x)
    x = np.expand_dims(x, 0)
    
    images = np.vstack([x])
    images = model.predict(images)
    images = tf.nn.softmax(images[0])

    pred = np.argmax(images)

    if pred == 0:
        desc = 'BLAST'
    elif pred == 1:
        desc = 'BLIGHT'
    elif pred == 2:
        desc = 'TUNGRO'
    
    classification = '%s' % (desc)
    return render_template('index.html', prediction=classification, image=imagepath)

if __name__ == '__main__':
    app.run(debug=True)