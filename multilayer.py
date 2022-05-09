import pandas as pd
from keras.layers import Dense, Input
from keras.models import Model
from keras.utils.vis_utils import plot_model
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib

# matplotlib.use('TkAgg')

"""
Data preprocessing
"""
# Load Data -- Numpy Implementation
# dataset = loadtxt('data/diabetes.csv', delimiter=',')
# x = dataset[:, 0:8]  # input values
# y = dataset[:, 8]  # output values

# dataset = loadtxt('data/diabetes.csv', delimiter=',')
# x = dataset[:, :-1]  # input values -- this will select all columns expect the last column
# y = dataset[:, -1]  # output values -- this means select only the last column

# Load data using pandas
"""
If dataset doesn't have headers. Use code below:

column_names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
dataset = pd.read_csv('data/diabetes.csv', names=column_names)
"""
dataset = pd.read_csv('data/diabetes.csv')

print(f'Shape of the data: {dataset.shape}')  # show the shape of data

# Split data into train and test
train_data = dataset.sample(frac=0.7, random_state=25)
test_data = dataset.drop(train_data.index)

# Separate the independent (x) and target (y) variable on training data
train_X = train_data.drop(columns=['Outcome'], axis=-1)
train_y = train_data['Outcome']

# Separate the independent (x) and target (y) variable on testing data
test_X = test_data.drop(columns=['Outcome'], axis=-1)
test_y = test_data['Outcome']

"""
Define the Keras Model
We are going to have 3 hidden layers in our network.
The first layer have 12 nodes with the activation function as ReLU
The second layer have 8 nodes with the activation function as ReLU
The third layer have 4 nodes with the activation function as ReLU
The output layer has a single node with the activation function as Sigmoid

Network architecture: 8/24/16/8/4/1

Note: the visible layer is also known as the input layer
"""
visible_layer = Input(shape=(8,))
first_layer = Dense(24, activation='relu')(visible_layer)
second_layer = Dense(16, activation='relu')(first_layer)
third_layer = Dense(8, activation='relu')(second_layer)
fourth_layer = Dense(4, activation='relu')(third_layer)
output_layer = Dense(1, activation='sigmoid')(fourth_layer)
model = Model(inputs=visible_layer, outputs=output_layer)

model.summary()  # to view summary of model

plot_model(model, to_file='multilayer_architecture.png')  # plot graph of the model

# Compile Model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Fit Model
model.fit(train_X, train_y, epochs=500, batch_size=250, verbose=2)

# Evaluation
loss_function, accuracy = model.evaluate(train_X, train_y, verbose=2)
print(f'\nLoss Function: {loss_function}')
print(f'Training Accuracy: {accuracy * 100}')

# Make Predictions
predictions = (model.predict(test_X) > 0.5).astype(int)

# Show Results
X = test_X.values.tolist()
for i in range(30):
    if predictions[i] == test_y.iloc[i]:
        print(f'\n{X[i]} => {predictions[i]} --- Expected output => {test_y.iloc[i]}. - Correct')
    else:
        print(f'\n{X[i]} => {predictions[i]} --- Expected output => {test_y.iloc[i]}. - Incorrect')

"""
Plot a scatter graph to show the points
"""
# Plot the graph
"""
sns.lmplot(x='Glucose', y='Age', data=dataset, hue='Outcome', fit_reg=False)
fig = plt.gcf()
fig.set_size_inches(15, 10)
# plt.legend()
plt.show(block=True)
plt.interactive(False)
plt.savefig('diabetes_graph.pdf')
"""
data_graph = pd.read_csv('data/diabetes.csv')
point1 = data_graph[data_graph['Outcome'] == 0]
point2 = data_graph[data_graph['Outcome'] == 1]
plt.rcParams.update({'figure.figsize': (8, 6), 'figure.dpi': 100})
plt.scatter(point1['Glucose'], point1['Age'], color='green', marker='*', label='Outcome=Negative')
plt.scatter(point2['Glucose'], point2['Age'], color='red', marker='v', label='Outcome=Positive')
plt.title('Diabetes Data')
plt.xlabel('Glucose')
plt.ylabel('Age')
plt.legend(loc='best')
plt.show()
