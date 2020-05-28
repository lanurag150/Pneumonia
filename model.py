from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
classifier = Sequential()
classifier.add(Conv2D(32, (3, 3), input_shape = (64, 64, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))
classifier.add(Conv2D(32, (3, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))
classifier.add(Conv2D(32, (3, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))
classifier.add(Flatten())
classifier.add(Dense(units = 128, activation = 'relu'))
classifier.add(Dense(units = 1, activation = 'sigmoid'))
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)
train_set = train_datagen.flow_from_directory('dataset/trainset',
                                                 target_size = (64, 64),
                                                 batch_size = 32,
                                                 class_mode = 'binary')

test_set = test_datagen.flow_from_directory('dataset/testset',
                                            target_size = (64, 64),
                                            batch_size = 32,
                                            class_mode = 'binary')

#classifier.fit_generator(train_set,
#samples_per_epoch = 5000,
#nb_epoch = 8,
#validation_data = test_set,
# nb_val_samples = 2000)
#classifier.save('model1.h5')
import numpy as np
from keras.preprocessing import image
img = image.load_img('p.jpeg', target_size=(64,64))
img = image.img_to_array(img)
img = np.expand_dims(img, axis=0)
MODEL_PATH = 'models/model1.h5'
from keras.models import load_model
model = load_model(MODEL_PATH)
result=model.predict(img)
print(result[0][0])
if result[0][0] == 0:
    print("infected")
else:
    print("uninfected") 


