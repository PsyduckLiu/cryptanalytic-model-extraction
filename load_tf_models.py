import tensorflow as tf
import numpy as np

model_path = "models/100_10x2_1_Seed42.keras"
model = tf.keras.models.load_model(model_path)

A = []
B = []


for i in range(1, len(model.layers)):
    print(model.layers[i].get_weights())
    A.append(model.layers[i].get_weights()[0])
    B.append(model.layers[i].get_weights()[1])

params = [A,B]

np.save("models/42_100-10-10-1", np.asarray(params, dtype="object"))