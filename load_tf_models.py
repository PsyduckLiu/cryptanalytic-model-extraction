import tensorflow as tf
import numpy as np

model_path = "models/mnist784_8x2_1v2.keras"
model = tf.keras.models.load_model(model_path)

A = []
B = []


for i in range(1, len(model.layers)):
    print(model.layers[i].get_weights())
    A.append(model.layers[i].get_weights()[0])
    B.append(model.layers[i].get_weights()[1])

params = [A,B]

np.save("models/42_784-8-8-1", np.asarray(params, dtype="object"))