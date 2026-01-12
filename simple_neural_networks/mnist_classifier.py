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

###### Images have 28 by 28 pixels and have greyscale values that range from 0 to 255 before preprocessing

fig, axes = plt.subplots(1,10, figsize=(12,3))

for i in range (0,10):
  axes[i].imshow(trainloader.dataset[i][0].squeeze(),cmap='gray')
  axes[i].set_title(trainloader.dataset[i][1])
  axes[i].axis('off')
plt.show()

class NeuralNetwork(nn.Module):
  def __init__(self):
    super(NeuralNetwork,self).__init__()
    self.input=nn.Linear(28*28,128)
    self.hidden=nn.Linear(128,64)
    self.output=nn.Linear(64,10)

  def forward(self,x):
    #x is gonna be the vector of values in the neural network at each layer
    #first we flatten the input vector  
    x=x.view(-1,28*28)
    #Then we put the vector through the neuron activations of out of the input layer and then apply Relu to the signal going down each branch
    x=F.relu(self.input(x))
    #Then we do the same for the hidden layer. Prep the signal arriving at the hidden layer input to travel down each branch of the hidden layer and apply relu before sending it to the output layer input
    x=F.relu(self.hidden(x))
    #then we do the same for the output layer but with a softmax instead of a relu. The signals coming out of each node are made into the scores of a softmax and then sent to the output layer outputs
    x=F.log_softmax(self.output(x),dim=1)
    return x





model=NeuralNetwork()



####Ok time to train this thing

loss_function=nn.NLLLoss()
optimizer=optim.Adam(model.parameters(),lr=0.001)

epochs=5
#An epoch is one pass through your training data#
#Like rereading or revising something
for epoch in range(epochs):
  for images, labels in trainloader:
    optimizer.zero_grad()

    model_output=model(images)

    loss=loss_function(model_output,labels)

    loss.backward()
    #the optimiser actually takes in the model as an inout so it knows what to update



    optimizer.step()
  print(f'Epoch [{epoch+1}/{epochs}] Loss: {loss.item():.4f}')    




correct=0
total=0

with torch.no_grad():
  for images, labels in testloader:
    output=model(images)
    _, predicted=torch.max(output,1)
    total += labels.size(0)
    correct += (predicted==labels).sum().item()

print(f'accuracy: {correct} images out of {total} with {(correct/total)*100} % accuracy')    

