import cv2
import numpy as np
import os
import sys
from tensorflow.keras import layers,models
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

EPOCHS = 7
IMG_WIDTH = 30
IMG_HEIGHT = 30
NUM_CATEGORIES = 43
TEST_SIZE = 0.4


def main():

    # Check command-line arguments
    if len(sys.argv) not in [2, 3]:
        sys.exit("Usage: python traffic.py data_directory [model.h5]")

    # Get image arrays and labels for all image files
    images, labels = load_data(sys.argv[1])

    # Split data into training and testing sets
    labels = to_categorical(labels)
    x_train, x_test, y_train, y_test = train_test_split(
        np.array(images), np.array(labels), test_size=TEST_SIZE
    )

    # Get a compiled neural network
    model = get_model()

    # Fit model on training data
    model.fit(x_train, y_train, epochs=EPOCHS)

    # Evaluate neural network performance
    model.evaluate(x_test,  y_test, verbose=2)

    # Save model to file
    if len(sys.argv) == 3:
        filename = sys.argv[2]
        model.save(filename)
        print(f"Model saved to {filename}.")


def load_data(data_dir):
    """
    Load image data from directory `data_dir`.

    Assume `data_dir` has one directory named after each category, numbered
    0 through NUM_CATEGORIES - 1. Inside each category directory will be some
    number of image files.

    Return tuple `(images, labels)`. `images` should be a list of all
    of the images in the data directory, where each image is formatted as a
    numpy ndarray with dimensions IMG_WIDTH x IMG_HEIGHT x 3. `labels` should
    be a list of integer labels, representing the categories for each of the
    corresponding `images`.
    """
    data_dir = os.path.join(os.getcwd(),data_dir)
    images,labels = [],[]
    for i in range(NUM_CATEGORIES):
        ith_dir = os.path.join(data_dir,str(i))
        for img_no in os.listdir(ith_dir):
            img = cv2.resize(cv2.imread(os.path.join(ith_dir,img_no)),dsize=(IMG_HEIGHT,IMG_WIDTH),interpolation=cv2.INTER_CUBIC)
            images.append(img)
            labels.append(i)

        print(ith_dir)

    print(f"{len(images)} Loaded")

    return (images,labels)

def get_model():
    """
    Returns a compiled convolutional neural network model. Assume that the
    `input_shape` of the first layer is `(IMG_WIDTH, IMG_HEIGHT, 3)`.
    The output layer should have `NUM_CATEGORIES` units, one for each category.
    """
    model = models.Sequential()

    model.add(layers.Conv2D(128,(3,3),activation="relu",input_shape=(IMG_HEIGHT,IMG_WIDTH,3)))

    model.add(layers.Conv2D(64,(3,3),activation="relu"))
    model.add(layers.MaxPooling2D((2,2)))
    model.add(layers.Flatten())

    model.add(layers.Dense(46))

    # Output Layer
    history = model.add(layers.Dense(NUM_CATEGORIES,activation="softmax"))
    model.compile(optimizer="adam",loss="categorical_crossentropy",metrics=["accuracy"])

    return model

if __name__ == "__main__":
    main()
