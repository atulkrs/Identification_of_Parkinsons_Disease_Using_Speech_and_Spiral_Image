import pandas as pd
import numpy as np
import os
import cv2
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.applications import VGG16
from keras.models import Model
from sklearn.preprocessing import LabelEncoder
from keras.layers import Dense, GlobalAveragePooling2D
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report

# Read the CSV file
data = pd.read_csv('Imageinfo2.csv')

# Extract image paths and class labels
image_paths = data['path'].values
class_labels = data['label'].values

# Initialize empty lists for images and labels
images = []
labels = []

# Load images and labels
for image_path, class_label in zip(image_paths, class_labels):
    image = cv2.imread(image_path)
    image = cv2.resize(image, (224, 224))  # Resize the image to desired size
    images.append(image)
    labels.append(class_label)
    
LE = LabelEncoder()
labels = LE.fit_transform(labels)

# Convert the lists to numpy arrays
images = np.array(images)
labels = np.array(labels)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

# Normalize the pixel values to the range [0, 1]
X_train = X_train.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0

# Convert class labels to categorical (one-hot encoding)
num_classes = len(np.unique(labels))
y_train = to_categorical(y_train, num_classes)
y_test = to_categorical(y_test, num_classes)

base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(256, activation='relu')(x)
predictions = Dense(num_classes, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

for layer in base_model.layers:
    layer.trainable = False

model.summary()

datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

datagen.fit(X_train)

model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

history = model.fit(datagen.flow(X_train, y_train, batch_size=32),
                    steps_per_epoch=len(X_train) / 32,
                    epochs=10,
                    validation_data=(X_test, y_test))

model.save('CNNModel.h5')

# Evaluate the model on test data
loss, accuracy = model.evaluate(X_test, y_test)
print('Test Loss:', loss)
print('Test Accuracy:', accuracy)

# Generate predictions for test data
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_test_classes = np.argmax(y_test, axis=1)

# Classification Report
print(classification_report(y_test_classes, y_pred_classes))

# Calculate Sensitivity (True Positive Rate)
cm = confusion_matrix(y_test_classes, y_pred_classes)
tn, fp, fn, tp = cm.ravel()
sensitivity = tp / (tp + fn)
print('Sensitivity:', sensitivity)

# Plot confusion matrix
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()
