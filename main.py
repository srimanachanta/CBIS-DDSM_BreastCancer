import os
import tensorflow as tf
from keras import layers, models
from keras.preprocessing.image import ImageDataGenerator
from tqdm import tqdm
import cv2
import numpy as np
from sklearn.model_selection import train_test_split


def main():
    imageSize = (256, 256)
    batchSize = 32

    categories = ["benign", "malignant"]

    images = []
    labels = []

    for cat in categories:
        print(f"Loading images from the '{cat.upper()}' category")
        basePath = os.path.join(r"Dataset", cat)
        for fileName in tqdm(os.listdir(basePath)):
            img = cv2.imread(os.path.join(basePath, fileName), 0)
            img = cv2.resize(img, imageSize)
            images.append(img)
            labels.append(cat)

    images = np.array(images) / 255.0
    labels = tf.keras.utils.to_categorical([categories.index(i) for i in np.array(labels)])

    trainImages, testImages, trainLabels, testLabels = train_test_split(images, labels, test_size=0.2)

    data_gen = ImageDataGenerator(
        rotation_range=15,
        shear_range=0.1,
        zoom_range=0.1,
        width_shift_range=0.15,
        height_shift_range=0.15,
        horizontal_flip=True
    )

    trainImages = trainImages.reshape((2452, 256, 256, 1))

    data_gen.fit(trainImages)

    model = models.Sequential(layers=[
        layers.Conv2D(filters=32, kernel_size=4, strides=3, padding='same', activation='relu', input_shape=(256, 256, 1)),
        layers.BatchNormalization(),
        layers.Conv2D(filters=64, kernel_size=4, strides=3, padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.Conv2D(filters=128, kernel_size=4, strides=3, padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.Conv2D(filters=256, kernel_size=4, strides=3, padding='same', activation='relu'),
        layers.BatchNormalization(),

        layers.Flatten(),

        layers.Dense(256, activation='relu'),
        layers.Dense(128, activation='relu'),
        layers.Dense(64, activation='relu'),

        layers.Dense(2, activation='sigmoid')
    ])

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    model.fit(
        trainImages,
        trainLabels,
        validation_data=(testImages, testLabels),
        epochs=50,
        batch_size=batchSize
    )


if __name__ == "__main__":
    main()
