import numpy as np
import pandas as pd
import os
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Dropout



data_dir = '/kaggle/input/gtsrb-german-traffic-sign'
train_path = os.path.join(data_dir, 'Train')
test_path = os.path.join(data_dir, 'Test')

IMG_HEIGHT = 30
IMG_WIDTH = 30
channels = 3
NUM_CATEGORIES = 43

print("Loading training data... this may take a minute.")
data = []
labels = []

for i in range(NUM_CATEGORIES):
    path = os.path.join(train_path, str(i))
    images = os.listdir(path)

    for img in images:
        try:
            image = Image.open(os.path.join(path, img))
            image = image.resize((IMG_HEIGHT, IMG_WIDTH))
            image = np.array(image)

            data.append(image)
            labels.append(i)
        except:
            print("Error loading image")

data = np.array(data)
labels = np.array(labels)

print("Data loaded!")
print(f"Total images: {data.shape[0]}, Image shape: {data.shape[1:]}")




'''
Split the data using test_train_split into X_train, X_val, y_train, y_val and normalize the data
'''


X_train, X_val, y_train, y_val = train_test_split(data,labels,test_size=0.2)
X_train =X_train/255.0
X_val =X_val/255.0

num_classes = 43 
y_train = to_categorical(y_train, 43)
y_val = to_categorical(y_val, 43)
print(f"Pre-processing complete. X_train shape: {X_train.shape}")

# TODO: One-hot encode the labels "y_train and y_val"




# Write the CNN code below with Maxpool layer, dropout rate, flatten it
model = Sequential([
    # First Convolutional Block
    Conv2D(filters=32, kernel_size=(5,5), activation='relu', input_shape=X_train.shape[1:]),
    Conv2D(filters=32, kernel_size=(5,5), activation='relu'),
    MaxPool2D(pool_size=(2, 2)),
    Dropout(rate=0.25),

    # Second Convolutional Block
    Conv2D(filters=64, kernel_size=(3, 3), activation='relu'),
    Conv2D(filters=64, kernel_size=(3, 3), activation='relu'),
    MaxPool2D(pool_size=(2, 2)),
    Dropout(rate=0.25),

    # Flattening and Dense Layers
    Flatten(),
    Dense(256, activation='relu'),
    Dropout(rate=0.5),
    Dense(num_classes, activation='softmax')
    
])

# After this compile the model using categorical_crossentropy loss, use adam optimizer and accuracy metrics
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()




# Decide epochs and batch_size which fits best for your model
epochs = 15
batch_size=64

history = model.fit(
    X_train, y_train,
    batch_size=batch_size,
    epochs=epochs, 
    validation_data=(X_val, y_val)
)

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Accuracy vs. Epochs')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss vs. Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

y_test = pd.read_csv(os.path.join(data_dir, 'Test.csv'))
labels_test = y_test["ClassId"].values
imgs = y_test["Path"].values
data_test = []

print("Loading test images...")
for img in imgs:
    image = Image.open(os.path.join(data_dir, img))
    image = image.resize((IMG_HEIGHT, IMG_WIDTH))
    data_test.append(np.array(image))

X_test = np.array(data_test)
X_test = X_test / 255.0

pred = model.predict(X_test)
pred_classes = np.argmax(pred, axis=1)
