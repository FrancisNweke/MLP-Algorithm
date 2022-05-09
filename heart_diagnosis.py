import pandas as pd
from keras.layers import Dense, Input, Dropout
from keras.models import Model
from keras.utils.vis_utils import plot_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression

# Data preprocessing
dataset = pd.read_csv('data/heart.csv')

print(dataset.shape)

# Separate the independent (x) and target (y) variable on testing data
dataset_X = dataset.drop(columns=['target'], axis=-1)
dataset_y = dataset['target']

# Split data into training and testing sets
train_X, test_X, train_y, test_y = train_test_split(dataset_X, dataset_y, random_state=42, test_size=0.2)

visible_layer = Input(shape=(13,))
fc_layer_1 = Dense(60, activation='tanh')(visible_layer)
fc_layer_2 = Dense(47, activation='tanh')(fc_layer_1)
fc_layer_3 = Dense(33, activation='tanh')(fc_layer_2)
dropout_layer = Dropout(rate=0.5)(fc_layer_3)
fc_layer_4 = Dense(15, activation='tanh')(dropout_layer)
fc_layer_5 = Dense(6, activation='tanh')(fc_layer_4)
output_layer = Dense(1, activation='sigmoid')(fc_layer_5)

model = Model(visible_layer, output_layer)

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(train_X, train_y, batch_size=310, epochs=500, verbose=2)

# model.save('data/model.h5')

loss_func, accuracy = model.evaluate(train_X, train_y, verbose=2)
print(f'\nLoss Function: {loss_func}\nAccuracy: {accuracy}')

prediction = (model.predict(test_X) > 0.5).astype(int)

print(accuracy_score(test_y, prediction))
