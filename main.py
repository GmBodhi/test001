from tensorflow.keras import datasets,layers,models,Sequential
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D
import matplotlib.pyplot as plt

(TrainIn, TrainOut), (TestIn, TestOut) = datasets.mnist.load_data()
TrainIn = TrainIn.reshape(60000,28,28,1)
TestIn = TestIn.reshape(10000,28,28,1)
TrainOut = to_categorical(TrainOut)
TestOut = to_categorical(TestOut)

Samkhya = Sequential()
#add Convolution  Layers
Samkhya.add(Conv2D(64, kernel_size=3, activation='relu', input_shape=(28,28,1),name='c1'))
Samkhya.add(MaxPooling2D(pool_size=(2, 2)))
Samkhya.add(Conv2D(64, kernel_size=3, activation='relu', input_shape=(28,28,1),name='c2'))
Samkhya.add(MaxPooling2D(pool_size=(2, 2)))


# Add Flatten/Dense Layers
Samkhya.add(Flatten(name='F1'))
Samkhya.add(Dense(10, activation='softmax',name='D1'))

Samkhya.compile(optimizer='sgd', loss='mse', metrics=['accuracy'])

print(Samkhya.summary())

TrainingLog=Samkhya.fit(TrainIn, TrainOut,epochs=6,validation_split=0.1)

import  PIL 
from PIL import  ImageEnhance , ImageOps  
from tensorflow.keras.preprocessing.image import img_to_array

Samkhya.evaluate(TestIn, TestOut)

plt.imshow(TestIn[150].reshape(28, 28),cmap='gray')

pred = Samkhya.predict(TestIn[150].reshape(1, 28, 28, 1))
print("The Samkhya predicted is",pred.argmax())

