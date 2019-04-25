
from keras.layers import Input, Dense
from keras.models import Model, Sequential
from sklearn import preprocessing
import numpy as np
import pandas as pd

# settings
encoding_dim = 5
inp_shape = 11
samplePercentSize = 1.0

# AUTOENCODER
inp = Input(shape=(inp_shape,))
# encoded representation of the input
encoded = Dense(encoding_dim, activation='relu')(inp)
# reconstruction of the input
decoded = Dense(inp_shape, activation='sigmoid')(encoded)

autoencoder = Model(inp, decoded)
# encoder section from the autoencoder
encoder = Model(inp, encoded)

# autoencoder compilation
autoencoder.compile(optimizer='adam', loss='mean_squared_error')

# Load data
data = np.loadtxt('../Data/winequality-white.csv', delimiter=';', skiprows=1)
EndRange = np.floor(len(data) * samplePercentSize)
(train, rest) = np.split(data, indices_or_sections=[int(EndRange)]) # get sample size
(x, y) = np.split(train, axis=1, indices_or_sections=[inp_shape])

# Adult income data processing
adult_income = pd.read_csv('../Data/adult_income.data', header=None)
(adult_income_x, adult_income_y) = np.split(adult_income, [14], axis=1)
adult_income_x = pd.get_dummies(adult_income_x, columns=[1,3,5,6,7,8,9,13]) # Normalize adult input data
adult_income_y.replace(to_replace=' <=50K', value=0, inplace=True)
adult_income_y.replace(to_replace=' >50K', value=1, inplace=True)

# Normalize input data
scaler = preprocessing.MinMaxScaler()
x = scaler.fit_transform(x) # x for winequality and heart disease, adult_income_x for adult income

# one hot encoding for output
oneHotEnc = preprocessing.OneHotEncoder(sparse=False) #TODO: MODIFIED
y_oneHotRep = oneHotEnc.fit_transform(y) #TODO: MODIFIED

# train autoencoder
autoencoder.fit(x, x,
                epochs=50,
                batch_size=32,
                shuffle=True)

# get learned features from encoder
learned_features = encoder.predict(x=x)

# CLASSIFIER
#classifier = Sequential() # Sequential Model
#classifier.add(Dense(encoding_dim * 2, input_dim=encoding_dim, activation='relu'))
#classifier.add(Dense(np.shape(train_y_oneHotRep)[1], activation='softmax'))
classificationLayer = Dense(np.shape(y_oneHotRep)[1], activation='softmax')(encoder.output)
classifier = Model(encoder.input, classificationLayer)

# compile classifier
classifier.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# train classifier and measure accuracy
classifier.fit(x, y_oneHotRep,
                epochs=250,
                batch_size=32,
                shuffle=True,
                validation_split=0.2)

# manual test accuracy
'''
results = classifier.predict(test_x)
match = 0
total = len(results)
for i in range(len(results)):
    maxIndexPredict = np.argmax(results[i])
    maxIndexTrue = np.argmax(test_y_oneHotRep[i])
    if maxIndexPredict == maxIndexTrue:
        match += 1

print('accuracy: ', match/total)
'''