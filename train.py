import numpy as np
from tensorflow import keras
from tensorflow.keras import layers, Sequential

(xtr, ytr), (xte, yte) = keras.datasets.cifar10.load_data()

def prep(x, y, a=3, b=5):
    idx = np.where((y==a) | (y==b))[0]
    x, y = x[idx]/255.0, y[idx]
    y = (y==b).astype("int32")
    return x, y

xtr, ytr = prep(xtr, ytr)
xte, yte = prep(xte, yte)

model = Sequential([
    layers.Conv2D(16,3,activation="relu",input_shape=(32,32,3)),
    layers.MaxPooling2D(),
    layers.Conv2D(32,3,activation="relu"),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(64,activation="relu"),
    layers.Dense(1,activation="sigmoid")
])
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
model.fit(xtr, ytr, epochs=5, batch_size=64, validation_split=0.1, verbose=2)
acc = model.evaluate(xte, yte, verbose=0)[1]
print(f"Test accuracy: {acc:.3f}")
model.save("cat_dog_cnn.keras")
