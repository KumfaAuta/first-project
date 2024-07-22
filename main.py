from flask import Flask, request, render_template, redirect, send_file
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from gtts import gTTS, gTTSError
import pyttsx3
import numpy as np
import os

app = Flask(__name__)

# Load your trained model
model = load_model('currency_classifier_model.h5')

# Define allowed extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Function to use pyttsx3 as a fallback
def text_to_speech_offline(text, file_path):
    engine = pyttsx3.init()
    engine.save_to_file(text, file_path)
    engine.runAndWait()

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join('uploads', filename)
            file.save(file_path)

            # Preprocess the image
            img = image.load_img(file_path, target_size=(128, 128))  # Adjust size as per your model
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
            img_array /= 255.0  # Normalize if required by your model

            # Predict the class
            prediction = model.predict(img_array)
            class_index = np.argmax(prediction)

            # Define class names
            class_names = [
                'Five naira', 
                'Ten naira', 
                'Twenty naira', 
                'Fifty naira', 
                'Hundred naira', 
                'Two hundred naira', 
                'Five hundred naira', 
                'One thousand naira'
            ]

            # Get the predicted class name
            predicted_class = class_names[class_index] if class_index < len(class_names) else "Unknown"

            # Convert text to speech using gTTS, with a fallback to pyttsx3
            audio_file = os.path.join('uploads', 'result.mp3')
            try:
                tts = gTTS(text=f'{predicted_class}', lang='en')
                tts.save(audio_file)
            except gTTSError:
                text_to_speech_offline(f'{predicted_class}', audio_file)

            return render_template('result.html', predicted_class=predicted_class, audio_file='uploads/result.mp3')
    
    return render_template('index.html')

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_file(os.path.join('uploads', filename))

if __name__ == '__main__':
    # Create uploads folder if it doesn't exist
    if not os.path.exists('uploads'):
        os.makedirs('uploads')
    app.run(debug=True)
