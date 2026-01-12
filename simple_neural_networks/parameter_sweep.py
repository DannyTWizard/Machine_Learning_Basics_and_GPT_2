#######You can honestly just use common sense about what parameters to tune
#######You can tune two parameters at a time, then use the optimal combination to
#######tune the next pair



import torch
import torchvision
import torchvision.transforms as transforms


# Define a transform to normalize the data
transform = transforms.Compose([transforms.ToTensor(),
                               transforms.Normalize((0.5,), (0.5,))])

# Load the training and test datasets
trainset = torchvision.datasets.MNIST(root='./data', train=True,
                                        download=True, transform=transform)

testset = torchvision.datasets.MNIST(root='./data', train=False,
                                       download=True, transform=transform)

# Create data loaders
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64,
                                          shuffle=True)

testloader = torch.utils.data.DataLoader(testset, batch_size=64,
                                         shuffle=False)

# Display the number of samples in each dataset
print("Number of samples in training set:", len(trainset))
print("Number of samples in test set:", len(testset))


#####Import your dependencies

import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F
import pandas as pd


#####Now I want to increase the accuracy as much as possible

#####Find out if you need to do batch norms

##### Find out if you need to do regularisation.

##### redefine the class to do a parameter sweep

#learning rate
#regularisation lamda

#number of hidden layers
#number of nodes per hidden layer




class NeuralNetwork(nn.Module):
  def __init__(self,input_size, output_size, layers,layer_size,hasnorm):

    super(NeuralNetwork,self).__init__()

    self.input_size=input_size
    self.output_size=output_size
    self.layers=layers
    self.layer_size=layer_size
    self.hasnorm=hasnorm

    self.input=nn.Linear(self.input_size,self.layer_size)

    if hasnorm:
      self.norm_layer=nn.BatchNorm1d(self.layer_size)

    self.hidden_layers=nn.ModuleList([nn.Linear(self.layer_size,self.layer_size) for _ in range(self.layers)])
    self.output=nn.Linear(self.layer_size,self.output_size)


  def forward(self,x):
    #x is gonna be the vector of values in the neural network at each layer
    #first we flatten the input vector
    x=x.view(-1,self.input_size)
    #Then we put the vector through the neuron activations of out of the input layer and then apply Relu to the signal going down each branch
    x=F.relu(self.input(x))
    x=self.norm_layer(x)
    #Then we do the same for the hidden layer. Prep the signal arriving at the hidden layer input to travel down each branch of the hidden layer and apply relu before sending it to the output layer input
    for layer in self.hidden_layers:
      x=F.relu(layer(x))
    #Then we do the same for the hidden layer. Prep the signal arriving at the hidden layer input to travel down each branch of the hidden layer and apply relu before sending it to the output layer input

    #then we do the same for the output layer but with a softmax instead of a relu. The signals coming out of each node are made into the scores of a softmax and then sent to the output layer outputs
    x=F.log_softmax(self.output(x),dim=1)
    return x


####I want a range of parameters
#lr 0.1 to 0.0001 (4)
#reg 1e-5 to 1e-2 (4)
#layers 1 to 20 (5)
#layer_size 16 to 256 (6)


loss_function=nn.NLLLoss()
optimizer=optim.Adam(model.parameters(),lr=0.001,weight_decay=1e-4)
epochs=5

layer_sizes=np.linspace(16,256,6,dtype=int)
layer_nums=np.array([1,5,10,15,20])

print(layer_sizes)

model_sweep_results=np.zeros((6,5))

print(model_sweep_results)



for i in range(0,len(layer_sizes)):
  for j in range(0,len(layer_nums)):
    model=NeuralNetwork(28*28,10,layer_nums[j],layer_sizes[i],True)

    for epoch in range(epochs):
      print(f'training an epoch for {layer_nums[j]}, layers with {layer_sizes[i]} nodes')
      for images, labels in trainloader:
        optimizer.zero_grad()

        model_output=model(images)

        loss=loss_function(model_output,labels)

        loss.backward()
        #the optimiser actually takes in the model as an inout so it knows what to update

        optimizer.step()

    model.eval()

    correct=0
    total=0

    with torch.no_grad():
      for images, labels in testloader:
        output=model(images)
        _, predicted=torch.max(output,1)
        total += labels.size(0)
        correct += (predicted==labels).sum().item()
    accuracy=(correct/total)
    print('accuracy=',accuracy)
    model_sweep_results[i,j]=accuracy

