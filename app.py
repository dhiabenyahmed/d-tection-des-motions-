from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

app = Flask(__name__)

# Load your grayscale and RGB models
grayscale_model = load_model("path/to/grayscale/model.h5")
rgb_model = load_model("path/to/rgb/model.h5")

# Function to preprocess image and get predictions
def process_image(image_path):
    # Load and preprocess the image based on your requirements
    # ...

    # Get predictions from the grayscale and RGB models
    prediction_grayscale = grayscale_model.predict(processed_image)
    prediction_rgb = rgb_model.predict(processed_image)

    # Combine predictions by taking the average
    combined_prediction = (prediction_grayscale + prediction_rgb) / 2

    # Get the predicted class index
    predicted_class_index = np.argmax(combined_prediction)

    return predicted_class_index

# Main route for the web interface
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Process the uploaded image and get the prediction
        uploaded_file = request.files['file']
        if uploaded_file.filename != '':
            image_path = 'path/to/temporary/folder/' + uploaded_file.filename
            uploaded_file.save(image_path)

            # Get the prediction for the image
            prediction_index = process_image(image_path)

            # Render the result to display on the page
            return render_template('index.html', prediction_index=prediction_index, image_path=image_path)

    return render_template('index.html', prediction_index=None, image_path=None)

if __name__ == '__main__':
    app.run(debug=True)
