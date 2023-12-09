import os  
import numpy as np 
import cv2 
from tensorflow.keras.utils import to_categorical
import tensorflow as tf

from keras.layers import Input, Dense 
from keras.models import Model
 
is_init = False
size = -1

label = []
dictionary = {}
c = 0
emotions = ['happy', 'sad', 'angry', 'surprise']


for i in os.listdir():
	if i.split(".")[-1] == "npy" and not(i.split(".")[0] == "labels"):  
		if not(is_init):
			is_init = True 
			X = np.load(i)
			size = X.shape[0]
			y = np.array([i.split('.')[0]]*size).reshape(-1,1)
		else:
			X = np.concatenate((X, np.load(i)))
			y = np.concatenate((y, np.array([i.split('.')[0]]*size).reshape(-1,1)))
		
		for emotion in emotions:
			if emotion in i.split('.')[0]:
				label.append(emotion)
			
		# label.append(i.split('.')[0])
		dictionary[i.split('.')[0]] = c  
		c = c+1


for i in range(y.shape[0]):
	y[i, 0] = dictionary[y[i, 0]]
y = np.array(y, dtype="int32")

###  hello = 0 nope = 1 ---> [1,0] ... [0,1]

y = to_categorical(y)

X_new = X.copy()
y_new = y.copy()
counter = 0 

cnt = np.arange(X.shape[0])
np.random.shuffle(cnt)

for i in cnt: 
	X_new[counter] = X[i]
	y_new[counter] = y[i]
	counter = counter + 1


ip = Input(shape=(X.shape[1]))



m = Dense(512, activation="relu")(ip)
m = Dense(256, activation="relu")(m)
m = Dense(256, activation="relu")(m)
m = Dense(256, activation="relu")(m)
m = Dense(256, activation="relu")(m)
m = Dense(256, activation="relu")(m)
m = Dense(256, activation="relu")(m)
m = Dense(256, activation="relu")(m)
m = Dense(256, activation="relu")(m)
m = Dense(256, activation="relu")(m)
m = Dense(256, activation="relu")(m)

checkpoint_cb = tf.keras.callbacks.ModelCheckpoint("model.h5", save_best_only=True, monitor='loss')
early_stopping_cb = tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True, monitor='loss')

op = Dense(y.shape[1], activation="softmax")(m) 

model = Model(inputs=ip, outputs=op)

model.compile(optimizer='rmsprop', loss="categorical_crossentropy", metrics=['acc'])

model.fit(X, y, epochs=5000, callbacks=[checkpoint_cb, early_stopping_cb], verbose=1)


model.save("modelsave.h5")
np.save("labels.npy", np.array(label))