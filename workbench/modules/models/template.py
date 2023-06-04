from keras import layers, models
from keras.optimizers import Adam
import pandas as pd


def build(input_shape) -> any:
    model = models.Sequential()

    # Section 1:
    model.add(
        layers.Conv2D(
            64, (3, 3), activation="relu", input_shape=input_shape, padding="same"
        )
    )
    model.add(layers.Conv2D(64, (3, 3), activation="relu", padding="same"))
    model.add(layers.MaxPooling2D((2, 2), strides=(2, 2)))

    # Section 2:
    model.add(
        layers.Conv2D(
            128, (3, 3), activation="relu", input_shape=input_shape, padding="same"
        )
    )
    model.add(layers.Conv2D(128, (3, 3), activation="relu", padding="same"))
    model.add(layers.MaxPooling2D((2, 2), strides=(2, 2)))

    # Section 3:
    model.add(layers.Conv2D(256, (3, 3), activation="relu", padding="same"))
    model.add(layers.Conv2D(256, (3, 3), activation="relu", padding="same"))
    model.add(layers.Conv2D(256, (3, 3), activation="relu", padding="same"))
    model.add(layers.Conv2D(256, (3, 3), activation="relu", padding="same"))
    model.add(layers.MaxPooling2D((2, 2), strides=(2, 2)))

    # Section 4:
    model.add(layers.Conv2D(512, (3, 3), activation="relu", padding="same"))
    model.add(layers.Conv2D(512, (3, 3), activation="relu", padding="same"))
    model.add(layers.Conv2D(512, (3, 3), activation="relu", padding="same"))
    model.add(layers.Conv2D(512, (3, 3), activation="relu", padding="same"))
    model.add(layers.MaxPooling2D((2, 2), strides=(2, 2)))

    # Section 5
    model.add(layers.Conv2D(512, (3, 3), activation="relu", padding="same"))
    model.add(layers.Conv2D(512, (3, 3), activation="relu", padding="same"))
    model.add(layers.Conv2D(512, (3, 3), activation="relu", padding="same"))
    model.add(layers.Conv2D(512, (3, 3), activation="relu", padding="same"))
    model.add(layers.MaxPooling2D((2, 2), strides=(2, 2)))

    # Classification Section
    model.add(layers.Flatten())
    model.add(layers.Dense(4096, activation="relu"))
    model.add(layers.Dropout(0.5))

    model.add(layers.Dense(4096, activation="relu"))
    model.add(layers.Dropout(0.25))

    model.add(layers.Dense(4, activation="softmax"))

    optimizer = Adam(learning_rate=0.0001)
    model.compile(
        optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"]
    )

    return model


def load_training_data(path: str) -> dict:
    df = pd.read_csv(path)
    train_label = df["emotion"]
    train_data = df.drop("emotion", axis=1)
    return {"data": train_data, "label": train_label}


def load_test_data(path: str) -> dict:
    df = pd.read_csv(path)
    test_label = df["emotion"]
    test_data = df.drop("emotion", axis=1)
    return {"data": test_data, "label": test_label}
