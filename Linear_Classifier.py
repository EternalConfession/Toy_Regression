from __future__ import print_function
import numpy as np
import sys
import os
from cntk import *
import cntk

#Set Device.
cntk.device.try_set_default_device(cntk.device.gpu(0))

#Generate Input Data.
input_dim = 2
num_output_classes = 2

np.random.seed(0)
# Helper Function for Data Generating
def generateRandomData(sample_size, input_dim, output_class):
    Y = np.random.randint(size=(sample_size, 1), low=0, high=output_class)
    X = np.random.randn(sample_size, input_dim)
    X = (X + 3) * (Y + 1)
    X = X.astype(np.float32)
    #Change the Data in Y into vector [1, 0, ... ,0] & [0, 1 ...,0]
    class_ind = [Y==class_number for class_number in range(output_class)]
    Y = np.asarray(np.hstack(class_ind), dtype=np.float32)
    return X,Y

#Generate Data
my_data_size = 256
features, labels = generateRandomData(my_data_size, input_dim, num_output_classes)

#Plot the Data & Show
import matplotlib.pyplot as plt

colors = ['r' if l == 0 else 'b' for l in labels[:,0]]
plt.figure(0)
plt.scatter(features[:,0], features[:,1],c=colors)
plt.xlabel("Some X data")
plt.ylabel("Some Y data")
#plt.show()
plt.savefig('Plot_Of_Data')

#Building the simplest Logistic regression
#Define Neural Network
mydict = {"w":None, "b":None}
feature = input(input_dim, np.float32)

def linear_layer(input_var, output_dim):
    input_dim = input_var.shape[0]
    weight_param = parameter(shape=(input_dim, output_dim))
    bias_param = parameter(shape=output_dim)

    mydict['w'], mydict['b'] = weight_param, bias_param

    return times(input_var, weight_param) + bias_param

output_dim = num_output_classes
z = linear_layer(feature, output_dim)

#Define Loss
label = input((num_output_classes), np.float32)
loss = cross_entropy_with_softmax(z,label)

eval_error = classification_error(z,label)

#Configure Training.
learning_rate = 0.5
lr_schedule = learning_rate_schedule(learning_rate, UnitType.minibatch)
learner = sgd(z.parameters, lr_schedule)
trainer = Trainer(z, (loss, eval_error), [learner])


#Some help functions
def moving_average(a, w=10):
    if len(a) < w: 
        return a[:]    
    return [val if idx < w else sum(a[(idx-w):idx])/w for idx, val in enumerate(a)]


# Defines a utility that prints the training progress
def print_training_progress(trainer, mb, frequency, verbose=1):
    training_loss, eval_error = "NA", "NA"

    if mb % frequency == 0:
        training_loss = trainer.previous_minibatch_loss_average
        eval_error = trainer.previous_minibatch_evaluation_average
        if verbose: 
            print ("Minibatch: {0}, Loss: {1:.4f}, Error: {2:.2f}".format(mb, training_loss, eval_error))
        
    return mb, training_loss, eval_error

#Run the Trainer && Feed Data
minibatch_size = 25
num_samples_to_train = 20000
num_minibatches_to_train = int(num_samples_to_train  / minibatch_size)

training_progress_output_freq = 50

plotdata = {"batchsize":[], "loss":[], "error":[]}

for i in range(0, num_minibatches_to_train):
    features, labels = generateRandomData(minibatch_size, input_dim, num_output_classes)
    
    # Specify input variables mapping in the model to actual minibatch data to be trained with
    trainer.train_minibatch({feature : features, label : labels})
    batchsize, loss, error = print_training_progress(trainer, i, 
                                                     training_progress_output_freq, verbose=1)
    
    if not (loss == "NA" or error =="NA"):
        plotdata["batchsize"].append(batchsize)
        plotdata["loss"].append(loss)
        plotdata["error"].append(error)
    
print("Training Done")

#Plot the loss & error
plotdata["avgloss"] = moving_average(plotdata["loss"])
plotdata["avgerror"] = moving_average(plotdata["error"])

# Plot the training loss and the training error

plt.figure(1)
plt.subplot(211)
plt.plot(plotdata["batchsize"], plotdata["avgloss"], 'b--')
plt.xlabel('Minibatch number')
plt.ylabel('Loss')
plt.title('Minibatch run vs. Training loss')

plt.savefig("Avg_loss")

plt.subplot(212)
plt.plot(plotdata["batchsize"], plotdata["avgerror"], 'r--')
plt.xlabel('Minibatch number')
plt.ylabel('Label Prediction Error')
plt.title('Minibatch run vs. Label Prediction Error')
plt.savefig("Avg_Error")

#Show the Result
weight_matrix = mydict['w'].value
bias_vec = mydict['b'].value
colors = ['r' if l == 0 else 'g' for l in labels[:,0]]
plt.figure()
plt.scatter(features[:,0],features[:,1],c = colors)
plt.plot([0, bias_vec[0]/weight_matrix[0][1]], [ bias_vec[1]/weight_matrix[0][0], 0], c = 'g', lw = 3)
plt.xlabel("X data")
plt.ylabel("y data")
plt.show()