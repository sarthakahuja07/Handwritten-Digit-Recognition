import os
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


# Decide if to load an existing model or to train a new one

if not os.path.isfile('handwritten_digits.h5'):
    # Loading the MNIST data set with samples and splitting it
    mnist = tf.keras.datasets.mnist
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    # Normalizing the data (making length = 1)
    # so that all values are within the range of 0 and 1.
    X_train = tf.keras.utils.normalize(X_train, axis=1)
    X_test = tf.keras.utils.normalize(X_test, axis=1)

    input_shape = (28, 28, 1)

    # Create a neural network model
    # Add two convolution layers and a max pooling layer
    # Add one flattened input layer for the pixels
    # Add one dense output layer for the 10 digits
    # Add one dense input layer for the flattened input 

    model = tf.keras.models.Sequential() #linear stack of layers
    model.add(tf.keras.layers.Conv2D(32, kernel_size=(3, 3),
              activation='relu', input_shape=input_shape)) #convolution layer with 32 filters of size 3x3 and relu activation function 
              #relu - rectified linear unit y = max(0, x) fast multiclass classification
    model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu')) #convolution layer with 64 filters of size 3x3 and relu activation function
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2))) #max pooling layer with 2x2 pool size to downsample the input
    model.add(tf.keras.layers.Dropout(0.25)) #dropout layer to avoid overfitting
    model.add(tf.keras.layers.Flatten()) # Flatten the input to 1D
    model.add(tf.keras.layers.Dense(units=128, activation=tf.nn.relu)) #dense layer with 128 neurons and relu activation function 
    model.add(tf.keras.layers.Dropout(0.5)) #dropout layer to avoid overfitting
    model.add(tf.keras.layers.Dense(units=10, activation=tf.nn.softmax)) #dense layer with 10 neurons and softmax activation function for output
    #Softmax converts a vector of values to a probability distribution. multiclass classification
    


# Compiling and optimizing model
    model.compile(optimizer='adam', #adam optimizer learning rate decay multiclass classification 
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])  # compile model with adam optimizer, sparse_categorical_crossentropy loss function and accuracy metric
                #sparse categorical crossentropy is used for multiclass classification 
               

    # fitting the model
    model.fit(X_train, y_train, 
            epochs=3,
            verbose=1,
            validation_data=(X_test, y_test)) #Data on which to evaluate the loss 

    # Evaluating the model
    val_loss, val_acc = model.evaluate(X_test, y_test)
    print(val_loss)
    print(val_acc)

    # Saving the model
    model.save('handwritten_digits.h5')
else:
    # Load the model
    model = tf.keras.models.load_model('handwritten_digits.h5')


# Load custom images and predict them
image_number = 1
while os.path.isfile('digits/digit{}.png'.format(image_number)):
    try:
        img = cv2.imread('digits/digit{}.png'.format(image_number))[:, :, 0] #read image and convert to grayscale 
        img = np.invert(np.array([img])) #invert image to make black background and white digits
        prediction = model.predict(img) #predict image using the model
        print("The number is {}".format(np.argmax(prediction))) #print the predicted number 
        plt.imshow(img[0], cmap=plt.cm.binary) #show image 
        plt.show() 
        image_number += 1
    except:
        print("Error reading image! Proceeding with next image...")
        image_number += 1
