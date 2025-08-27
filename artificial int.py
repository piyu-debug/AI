import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt

#load mnist dataset
(x_train,y_train),(x_test,y_test)=tf.keras.datasets.mnist.load_data()

#normaize the data

x_train,x_test = x_train/255.0, x_test/255.0

#build the model
model = models.Sequential([
    layers.Flatten(input_shape=(28,28)),
    layers.Dense(128,activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(10,activation='softmax')
])

#compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

#train the model

model.fit(x_train,y_train,epochs=5)

#evaluate the model

test_loss, test_acc = model.evaluate(x_test, y_test)

print(f"Test accuracy: {test_acc}")

#make predictions

predictions = model.predict(x_test)

#Display the First Image and Prediction

plt.imshow(x_test[0], cmap=plt.cm.binary)

plt.title(f"Predicted: {predictions[0].argmax()}")

plt.show()


