# -*- coding: utf-8 -*-
"""
Created on Mon Feb 24 21:56:06 2020

@author: aopre
"""

from keras.layers import MaxPooling2D, Flatten, Conv2D, Dense
from keras.models import Sequential

classifier = Sequential()

classifier.add(Conv2D(128, (5, 5), input_shape=(64, 64, 3), activation='relu', padding='same'))
classifier.add(MaxPooling2D(pool_size=(2, 2), padding='same'))

classifier.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
classifier.add(MaxPooling2D(pool_size=(2, 2), padding='same'))

classifier.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
classifier.add(MaxPooling2D(pool_size=(2, 2), padding='same'))

classifier.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
classifier.add(MaxPooling2D(pool_size=(2, 2), padding='same'))

classifier.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
classifier.add(MaxPooling2D(pool_size=(2, 2), padding='same'))

classifier.add(Flatten())

classifier.add(Dense(units=64, activation='relu'))
classifier.add(Dense(units=64, activation='relu'))

classifier.add(Dense(units=5, activation='softmax'))

classifier.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory("Diabetic Retinopathy\gaussian_filtered_images\Train_set",
                                                 target_size=(64, 64),
                                                 batch_size=10, 
                                                 class_mode='categorical')
test_set = test_datagen.flow_from_directory("Diabetic Retinopathy\gaussian_filtered_images\Test_set",
                                            target_size=(64, 64),
                                            batch_size=10, 
                                            class_mode='categorical')

classifier.fit_generator(training_set,
                         steps_per_epoch=2926,
                         epochs=10,
                         validation_data=test_set,
                         validation_steps=731,
                         workers=16,
                         max_queue_size=10)


from keras.preprocessing import image
import numpy as np
test_image = image.load_img("Diabetic Retinopathy\\gaussian_filtered_images\\Single_prediction\\severe\\3b018e8b7303.png", target_size=(64, 64, 3))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis=0)
result = classifier.predict(test_image)
training_set.class_indices