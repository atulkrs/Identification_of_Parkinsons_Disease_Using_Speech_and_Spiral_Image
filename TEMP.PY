import cv2
import numpy as np
from keras.models import load_model
from keras.preprocessing import image

# Load the trained model
model = load_model('RCNNModel.h5')

# Load and preprocess the test image
test_image_path = 'Severity\\healthy\\V01HE01.png'
img = cv2.imread(test_image_path)
img = cv2.resize(img, (224, 224))
img = img.astype('float32') / 255.0
img_array = np.expand_dims(img, axis=0)

# Perform prediction
prediction = model.predict(img_array)
predicted_class = np.argmax(prediction)

# Print the predicted class
print(f"Predicted class: {predicted_class}")
