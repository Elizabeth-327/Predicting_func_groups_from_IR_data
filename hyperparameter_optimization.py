"""Performs optimization of the hyper-parameters of the CNN. Written by Guwon Jung."""

import numpy as np
import pickle
import pandas as pd
import plaidml.keras
import os

plaidml.keras.install_backend()
os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"

from keras import backend as K
from keras.models import Model
from keras.layers import Input, MaxPooling1D, Dropout, Activation
from keras.layers import Conv1D, Dense, Flatten, BatchNormalization
from keras.optimizers import Adam
from keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint, Callback
from sklearn.metrics import f1_score, recall_score, precision_score
from skopt import gp_minimize
from skopt.space import Real,Integer
from skopt.utils import use_named_args
from skopt import callbacks
from skopt.callbacks import CheckpointSaver
from sklearn.utils import class_weight
#Define paths
results_dir = '../checkpoints/'
params_dir = '../searched_parameters/'

# Define search space
dim_num_dense_layers = Integer(low=1, high=4, name='num_dense_layers') #number of fully connected layers in the model
dim_num_filters = Integer(low=4, high=32, name='num_filters') #number of filters in the convolutional layers
dim_dense_divisor = Real(low=0.25, high=0.8, name='dense_divisor') #used to determine the size of subsequent fully connected layers
dim_num_cnn_layers = Integer(low=1, high=5, name='num_cnn_layers') #number of convolutional layers
dim_dropout = Real(low=0, high=0.5, name='dropout') #dropout rate to prevent overfitting
dim_batch_size = Integer(low=8, high=512, name='batch_size')
dim_kernel_size = Integer(low=2, high=12, name='kernel_size', dtype='int') #size of each filter
dim_num_dense_nodes = Integer(low=1000, high=5000, name='num_dense_nodes')

dimensions = [dim_num_dense_layers,
              dim_num_filters,
              dim_dense_divisor,
              dim_num_cnn_layers,
              dim_dropout,
              dim_batch_size,
              dim_kernel_size,
              dim_num_dense_nodes]

# Load training data
with open('processed_dataset.pickle', 'rb') as handle:
    dict_data = pickle.load(handle)

# Define training, validation, and test sets as tables
X_train = pd.DataFrame(dict_data['X_train_1'])
X_val = pd.DataFrame(dict_data['X_val_1'])
y_train = pd.DataFrame(dict_data['y_train_1'])
y_val = pd.DataFrame(dict_data['y_val_1'])
X_test = pd.DataFrame(dict_data['X_test'])
y_test = pd.DataFrame(dict_data['y_test'])

# Convert tables to arrays where each element is a 32-bit float
X_train = np.asarray(X_train).astype('float32')
X_test = np.asarray(X_test).astype('float32')
X_val = np.asarray(X_val).astype('float32')
y_train = np.asarray(y_train).astype('float32')
y_test = np.asarray(y_test).astype('float32')
y_val = np.asarray(y_val).astype('float32')

class_weights_list = []
for i in range(y_train.shape[1]):
    y_train_label = y_train[:, i]
    class_weights = class_weight.compute_class_weight(class_weight='balanced', classes=np.unique(y_train_label), y=y_train_label)
    class_weights_list.append(class_weights)

# Convert the list of class weights to a dictionary for use in training
class_weight_dict = {i: class_weights_list[i] for i in range(len(class_weights_list))}

print("Class Weights: ", class_weight_dict)

# Reshape to 3D arrays to be compatible with 1D convolutional layers
# Each element is now an array of length 1
X_train = X_train.reshape(X_train.shape[0], 600, 1)
X_val = X_val.reshape(X_val.shape[0], 600, 1)

# Define variables
input_shape = X_train.shape[1:] #each sample (input) has 600 features with 1 channel
num_classes = y_train.shape[1] #the number of response variables (functional groups)

# Fixed hyperparameters for CNN
maximum_epochs = 1000 
early_stop_epochs = 10 #training stops if the validation loss doesn't improve after this many epochs
learning_rate_epochs = 5 #learning rate is reduced if the validation loss doesn't improve after this many epochs

# Parameters that change for each iteration (update of model hyperparameters)
list_early_stop_epochs = []
list_validation_loss = []
list_saved_model_name = []

class Metrics(Callback):
    """Define loss function."""
    def __init__(self, validation):
        super(Metrics, self).__init__() #calls the initializer of the parent Callback class to ensure proper initialization
        self.validation = validation #stores validation data (features and targets)

    def on_train_begin(self, logs={}): 
        '''Called at the beginning of training.'''
        self.val_f1s = [] #to store F1 scores for each epoch
        self.val_recalls = [] #to store recall scores for each epoch
        self.val_precisions = [] #to store precision scores for each epoch

    def on_epoch_end(self, epoch, logs={}): #logs provides metrics and loss values for the current epoch
        '''Called at the end of each epoch during training.'''
        val_predict = (np.asarray(self.model.predict(self.validation[0]))).round() #predicted labels
        val_targ = self.validation[1] #true validation labels
        
        val_f1 = f1_score(val_targ, val_predict, average='micro') #calculates f1 score for validation set
        val_recall = recall_score(val_targ, val_predict, average='micro') #calculates recall for validation set
        val_precision = precision_score(val_targ, val_predict, average='micro') #calculates precision for validation set

        self.val_f1s.append(round(val_f1, 6))
        self.val_recalls.append(round(val_recall, 6))
        self.val_precisions.append(round(val_precision, 6))

        global f1score #why declare it as a global variable??
        f1score = val_f1 #updates global f1score with the current F1 score
        print(f'val_f1: {val_f1}, val_precision: {val_precision}, val_recall: {val_recall}')
        return #why type this?
    
def create_model(num_dense_layers, num_filters, dense_divisor, num_cnn_layers, dropout, kernel_size, num_dense_nodes):
    """Creates the architecture for a CNN model with given parameters."""
    # Create input tensor (the shape of the data that is fed into the neural network)
    input_tensor = Input(shape=input_shape)

    # Create first convolutional layer
    x = Conv1D(filters=num_filters, kernel_size=(kernel_size), strides=1, padding='same')(input_tensor) #applies a convolution operation to the input tensor
    x = BatchNormalization()(x) #normalizes the output of the convolution
    x = Activation('relu')(x) #applies the ReLU activation function to introduce non-linearity
    x = MaxPooling1D(pool_size=2, strides=2)(x) #applies a pooling operation to reduce the dimensionality of the input
    num_filters = num_filters * 2 #doubles the number of filters for the next convolutional layer

    # Create additional convolutional layers
    for layer in range(num_cnn_layers - 1):
        x = Conv1D(filters=num_filters, 
               kernel_size=(kernel_size), 
               strides=1, 
               padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = MaxPooling1D(pool_size=2, strides=2)(x)
        num_filters = num_filters * 2
    
    # Flatten the output of the convolutional layers
    x = Flatten()(x) 

    # Create first fully connected (dense) layer
    x = Dense(num_dense_nodes, activation='relu')(x)
    num_dense_nodes = int(num_dense_nodes * dense_divisor) #number of nodes is reduced by a factor of dense_divisor for the next layer
    x = Dropout(dropout)(x) #applies dropout to prevent overfitting

    # Create additional dense layers
    for i in range(num_dense_layers - 1):
        x = Dense(num_dense_nodes, activation='relu')(x)
        x = Dropout(dropout)(x)
        num_dense_nodes = int(num_dense_nodes * dense_divisor)
    
    # Create output tensor
    output_tensor = Dense(num_classes, activation='sigmoid')(x)
    
    # Instantiate model
    model = Model(inputs=input_tensor, outputs=output_tensor) #instantiate the model, now that you have your inputs AND outputs

    # Print model summary
    model.summary() #Prints summary of the model architecture (layers, output shapes, number of parameters, etc.)

    optimizer = Adam(lr=2.5e-4)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=[])

    return model

@use_named_args(dimensions=dimensions) 
def fitness(num_dense_layers,
            num_filters,
            dense_divisor,
            num_cnn_layers,
            dropout,
            batch_size,
            kernel_size,
            num_dense_nodes):
    """Defines the settings for the optimization of hyper-parameters."""
    # Print the chosen hyper-parameters for the epoch
    print('num_dense_layers:', num_dense_layers)
    print('num_filters:', num_filters)
    print('dense_divisor:', dense_divisor)
    print('num_cnn_layers:', num_cnn_layers)
    print('dropout:', dropout)
    print('batch_size:', batch_size)
    print('kernel_size', kernel_size)
    print('num_dense_nodes', num_dense_nodes)

    # Create model name and print
    model_name = 'cnn_' + str(np.random.uniform(0, 1, ))[2:9]
    print('model_name: ', model_name)

    # Create a CNN architecture with these hyperparameters
    model = create_model(num_dense_layers=num_dense_layers,
                      num_filters=num_filters,
                      dense_divisor=dense_divisor,
                      num_cnn_layers=num_cnn_layers,
                      dropout=dropout,
                      kernel_size=kernel_size,
                      num_dense_nodes=num_dense_nodes)
    
    # Create several callbacks that will be run during training
    callback_list = [EarlyStopping(monitor='val_loss', patience=early_stop_epochs),
                     ReduceLROnPlateau(monitor='val_loss', 
                                    factor=0.1, 
                                    patience=learning_rate_epochs, 
                                    verbose=1, 
                                    mode='auto', 
                                    min_lr=1e-6),
                     ModelCheckpoint(os.path.join(results_dir, model_name + '.h5'), 
                                  monitor='val_loss', 
                                  save_best_only=True), #saves model to file if it has the best performance on the validation set
                     Metrics(validation=(X_val, y_val))]
    
    # Use Keras to train the model
    history = model.fit(x=X_train,
                     y=y_train,
                     epochs=maximum_epochs,
                     batch_size=batch_size,
                     validation_data=(X_val, y_val),
                     callbacks=callback_list) #returns training metrics
    
    # Define validation loss
    val_loss = history.history['val_loss'][-1]

    # Record actual best epochs and validation loss
    list_early_stop_epochs.append(len(history.history['val_loss']) - early_stop_epochs)
    list_validation_loss.append(np.min(history.history['val_loss']))
    list_saved_model_name.append(model_name)

    # Delete the model with these hyperparameters from memory
    del model

    # Clear the Keras session
    K.clear_session()

    return val_loss

if not os.path.exists(results_dir):
    os.makedirs(results_dir)

# Apply Bayesian optimization to optimize the hyperparameters of the CNN

# Define checkpoints to save the progress of the optimization process
checkpoint_saver = CheckpointSaver(results_dir + 'checkpoint.pkl', compress=9)

# Run optimization, where model is being trained multiple times using a different set of hyperparameters each time
search_result = gp_minimize(func=fitness, dimensions=dimensions, acq_func='EI', n_calls=50, callback=[checkpoint_saver], n_jobs=-1) 

# Store search results in a dataframe
results_list = []
for result in zip(search_result.func_vals, #a list of the validation loss values for each set of hyperparameters
                  search_result.x_iters, #a list of hyperparameters used in each iteration
                  list_early_stop_epochs,
                  list_validation_loss,
                  list_saved_model_name): 
    temp_list = []
    temp_list.append(result[0])
    temp_list.append(result[2])
    temp_list.append(result[3])
    temp_list.append(result[4])
    temp_list = temp_list + result[1]
    results_list.append(temp_list)

# Define columns of the dataframe
df_results = pd.DataFrame(results_list, columns=['last_val_loss', 
                                                 'epochs', 
                                                 'lowest_val_loss', 
                                                 'model_name',
                                                 'num_dense_layers',
                                                 'num_filters', 
                                                 'dense_divisor', 
                                                 'num_cnn_layers', 
                                                 'dropout',
                                                 'batch_size',
                                                 'kernel_size',
                                                 'num_dense_nodes'])


# Save dataframe to pickle and csv files
df_results.to_pickle(params_dir + 'searched_parameters.pkl')
df_results.to_csv(params_dir + 'searched_parameters.csv')
