import torch
import torchvision
import torchvision.transforms as transforms
from mpl_toolkits.mplot3d import Axes3D  # For 3D plotting


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
                                          shuffle=True,
                                          pin_memory=True,
                                          num_workers=4)

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

import dataclasses
from dataclasses import dataclass


fig, axes = plt.subplots(1,10, figsize=(12,3))

for i in range (0,10):
  axes[i].imshow(trainloader.dataset[i][0].squeeze(),cmap='gray')
  axes[i].set_title(trainloader.dataset[i][1])
  axes[i].axis('off')
plt.show()

def initialize_spherical_surface(num_attractors, N, radius=1.0):
    x = torch.randn(num_attractors, N)
    norms = torch.norm(x, dim=1, keepdim=True)
    unit_vectors = x / norms
    return radius * unit_vectors

def initialize_ring_2d(num_attractors, radius=1.0):
    """
    Initialize points evenly distributed on a circle in 2D.
    
    Args:
        num_attractors: Number of attractors
        radius: Radius of the circle
    """
    # Evenly spaced angles
    angles = torch.linspace(0, 2 * np.pi, num_attractors, requires_grad=False)
    
    # Convert to Cartesian coordinates
    x = radius * torch.cos(angles)
    y = radius * torch.sin(angles)
    
    return torch.stack([x, y,torch.zeros(num_attractors)], dim=1)    

class CustomAttractorLayer(nn.Module):
    def __init__(self, N, num_attractors, potential_hidden_dimension,eta):
        super().__init__()
        self.N = N
        self.num_attractors = num_attractors
        self.potential_hidden_dimension = potential_hidden_dimension
        self.attractor_positions_tensor = nn.Parameter(torch.randn(num_attractors, N))
        self.eta = eta
        self.potential_gradient_in=nn.linear(1,potential_hidden_dimension)
        self.potential_gradient_out=nn.linear(potential_hidden_dimension,1)

        def potential_gradient(self,x_in):
            x=self.potential_gradient_in(x_in)
            x=self.gelu(x)
            x=self.potential_gradient_out(x)
            return x

        def forward(self,x):
            x_col = x.unsqueeze(1)
            concatenated_input = torch.cat([x_col, self.attractor_positions_tensor], dim=1)
            mask_shape=concatenated_input.shape[1]
            mask_tensor=torch.zeros(mask_shape)
            mask_tensor[0]=1
            distances_tensor=torch.zeros_like(concatenated_input)
            first_vector = concatenated_input[:, 0]  # Shape: (N,) 
    # Expand first_vector to match concatenated_input shape
            first_vector_expanded = first_vector.unsqueeze(1).expand_as(concatenated_input)
    # Now: (N, num_vectors) with first_vector repeated in each column    
            distances_tensor = first_vector_expanded - concatenated_input
            norms = torch.norm(distances_tensor, dim=0)
            potential_gradient=self.potential_gradient(norms)
            weights=torch.where(mask_tensor.bool(),0,potential_gradient/norms)
            A=weights
            D_0=A.sum(dim=0)
            L=torch.zeros_like(A)
            L[0]=D_0
            L=L-A
            force_components=concatenated_input*L
            force=force_components.sum(dim=0)
            x=x+self.eta*force
            return x














        


        


        
    def forward(self, x):
      x = self.encoder(x)
      x = self.decoder(x)

class one_particle_flap_layer(nn.Module):
    def __init__(self, N, num_attractors, potential_hidden_dimension, eta, pre_factor):
        super().__init__()
        self.N = N
        self.num_attractors = num_attractors
        self.potential_hidden_dimension = potential_hidden_dimension
        #self.attractor_positions_tensor = nn.Parameter(torch.randn(num_attractors, N))
        self.eta = eta
        # In __init__:
        #self.attractor_positions_tensor = nn.Parameter(initialize_spherical_surface(num_attractors, N, radius=1.0).to(device))
        self.attractor_positions_tensor = nn.Parameter(initialize_ring_2d(num_attractors, radius=1.0))
        self.pre_factor = pre_factor
        # Fixed: Capital L in Linear
        #self.potential_gradient_in = nn.Linear(num_attractors+1, potential_hidden_dimension)
        #self.potential_gradient_out = nn.Linear(potential_hidden_dimension, num_attractors+1)
        
        # Fixed: Capital L in Linear
        self.potential_gradient_in = nn.Linear(1, potential_hidden_dimension)
        self.potential_gradient_in_2 = nn.Linear(potential_hidden_dimension, potential_hidden_dimension)
        self.potential_gradient_out = nn.Linear(potential_hidden_dimension, 1)
        

        # Add GELU activation
        self.gelu = nn.GELU(approximate='tanh')

    def potential_gradient(self, x_in):
        original_shape = x_in.shape
        #print(f'original_shape: {original_shape}')
        x_flat = x_in.view(-1, 1)
        #print(f'x_flat shape: {x_flat.shape}')
        x = self.potential_gradient_in(x_flat)
        x = self.gelu(x)
        x = self.potential_gradient_in_2(x)
        x = self.gelu(x)
        x = self.potential_gradient_out(x)
        x = x.view(original_shape)
        return x

    def potential_gradient(self, x_in):
        original_shape = x_in.shape
        #print(f'original_shape: {original_shape}')
        x_flat = x_in.view(-1, 1)
        #print(f'x_flat shape: {x_flat.shape}')
        x=self.pre_factor*torch.ones_like(x_flat)
        x = x.view(original_shape)
        
        return x




                    

    def forward(self, x):
        #x_col = x.unsqueeze(1)
        batch_size = x.shape[0]
        attractors_expanded = self.attractor_positions_tensor.unsqueeze(0).expand(batch_size, -1, -1)
        x_col=x.unsqueeze(1)

        #print(f'x_col shape: {x_col.shape}')
        #print(f'attractor_positions_tensor shape: {self.attractor_positions_tensor.shape}')
        concatenated_input = torch.cat([x_col, attractors_expanded], dim=1)
        mask_shape = concatenated_input.shape[1]
        
        # Create mask (non-differentiable, but that's fine for a fixed mask)
        mask_tensor = torch.zeros(mask_shape, device=x.device, dtype=x.dtype)
        mask_tensor[0] = 1
        
        distances_tensor = torch.zeros_like(concatenated_input)
        first_vector = concatenated_input[:, 0]
        first_vector_expanded = first_vector.unsqueeze(1).expand_as(concatenated_input)
        distances_tensor = first_vector_expanded - concatenated_input
        norms = torch.norm(distances_tensor, dim=2)
        #print(f'norms shape: {norms.shape}')
        
        # Add small epsilon to avoid division by zero
        norms_safe = norms + 1e-8
        
        potential_gradients = self.potential_gradient(norms)

        weights = torch.where(mask_tensor.bool(), 0, potential_gradients / (norms_safe))
        A = weights
        D_0 = A.sum(dim=0)
        L = torch.zeros_like(A)
        L[0] = D_0  # This assignment preserves gradients
        L = L - A
        #print(f'L shape: {L.shape}')
        #print(f'concatenated_input shape: {concatenated_input.shape}')
        L_expanded=L.unsqueeze(-1)
        force_components = concatenated_input * L_expanded
        #print(f'force_components shape: {force_components.shape}')
        force = force_components.sum(dim=1)
        #print(f'force shape :{force.shape}')
        x = x + self.eta * force
        return x

class one_particle_flap_block(nn.Module):
  def __init__(self, d_in, d_out, N, num_attractors, num_attractor_layers, potential_hidden_dimension, eta, pre_factor, iterations):
    super().__init__()
    self.N = N
    self.num_attractors = num_attractors
    self.potential_hidden_dimension = potential_hidden_dimension
    self.eta = eta
    self.pre_factor = pre_factor
    self.iterations = iterations
    self.encoder = nn.Linear(d_in, N)
    self.decoder = nn.Linear(N, d_out)
    self.flap_layers = nn.ModuleList([one_particle_flap_layer(N, num_attractors, potential_hidden_dimension, eta, pre_factor) for _ in range(num_attractor_layers)])
    
  def forward(self, x):
    #print(f'x shape: {x.shape}')
    x=x.view(-1,28*28)
    #print(f'x shape after view: {x.shape}')
    x = F.gelu(self.encoder(x))
    #print(f'x shape after encoder: {x.shape}')
    for flap_layer in self.flap_layers:
      for _ in range(self.iterations):
        x=x.view(-1,self.N)
        x = flap_layer(x)
    x = F.log_softmax(self.decoder(x),dim=1)
    return x


# To:
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
torch.cuda.set_device(0)  # Explicitly set default GPU

print(f'Using device: {device}')

# Print GPU information if available
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'CUDA Version: {torch.version.cuda}')
    print(f'GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB')

model=one_particle_flap_block(28*28,10,3,30,1,1,2.3,1,1)
run='27'
model = model.to(device)  # Move all model parameters to GPU


####Generate computational graph automatically
def generate_computational_graph(model, input_shape=(1, 28, 28), output_file='computational_graph'):
    """
    Automatically generate and save the computational graph for a PyTorch module.
    
    Args:
        model: PyTorch nn.Module instance
        input_shape: Shape of input tensor (batch_size, channels, height, width) or (batch_size, features)
        output_file: Output filename (without extension) for the graph
    """
    try:
        from torchviz import make_dot
        
        # Create a dummy input with the specified shape
        dummy_input = torch.randn(1, *input_shape[1:]) if len(input_shape) > 1 else torch.randn(*input_shape)
        
        # Forward pass to generate the graph
        model.eval()
        output = model(dummy_input)
        
        # Generate the graph
        graph = make_dot(output, params=dict(list(model.named_parameters())))
        
        # Save as PNG
        graph.render(output_file, format='png', cleanup=True)
        print(f"Computational graph saved as {output_file}.png")
        
        # Also save as PDF
        graph.render(output_file, format='pdf', cleanup=True)
        print(f"Computational graph saved as {output_file}.pdf")
        
        return graph

    except ImportError:
        print("torchviz not installed. Installing it now...")
        import subprocess
        import sys
        subprocess.check_call([sys.executable, "-m", "pip", "install", "torchviz", "graphviz"])
        print("Please install graphviz system package:")
        print("  Ubuntu/Debian: sudo apt-get install graphviz")
        print("  macOS: brew install graphviz")
        print("  Windows: Download from https://graphviz.org/download/")
        print("\nTrying alternative method...")
        
        # Alternative: Use PyTorch's built-in visualization
        try:
            from torch.utils.tensorboard import SummaryWriter
            dummy_input = torch.randn(1, *input_shape[1:]) if len(input_shape) > 1 else torch.randn(*input_shape)
            writer = SummaryWriter(log_dir='./graph_logs')
            writer.add_graph(model, dummy_input)
            writer.close()
            print(f"Computational graph saved using TensorBoard. Run: tensorboard --logdir=./graph_logs")
        except Exception as e:
            print(f"Could not generate graph: {e}")

#total_params = sum(p.numel() for p in model.parameters())
#print(f"Total parameters: {total_params:,}")

#trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
#print(f"Trainable parameters: {trainable_params:,}")

# Generate the computational graph
#generate_computational_graph(model, input_shape=(1, 28, 28), output_file='flap_200_10_0.001_30_graph')


####Ok time to train this thing


#####No regularisation
#loss_function=nn.NLLLoss()
#optimizer=optim.Adam(model.parameters(),lr=0.001)


# Separate optimizers with different weight decay
loss_function=nn.NLLLoss()
attractor_params = []
encoder_decoder_params = []

for name, param in model.named_parameters():
    if 'encoder' in name or 'decoder' in name:
        encoder_decoder_params.append(param)
    else:
        attractor_params.append(param)

# Higher weight decay for encoder/decoder
optimizer = optim.Adam([
    {'params': encoder_decoder_params, 'weight_decay': 0.01},  # Regularize encoder/decoder
    {'params': attractor_params, 'weight_decay': 0.0}  # Don't regularize attractors
], lr=0.001)


#print(f"encoder_decoder_params:{encoder_decoder_params}")
#print(f"attractor_params:{attractor_params}")

encoder_decoder_count = sum(p.numel() for p in encoder_decoder_params)
attractor_count = sum(p.numel() for p in attractor_params)

#print(f"encoder_decoder_params: {encoder_decoder_count:,} parameters")
#print(f"attractor_params: {attractor_count:,} parameters")

#print(f"shapes:{len(encoder_decoder_params.shape)}{len(attractor_params.shape)}")

epochs=20
#An epoch is one pass through your training data#
#Like rereading or revising something

attractor_positions = model.flap_layers[0].attractor_positions_tensor.detach().cpu().numpy()
    # Shape: (num_attractors, 3)
    
    # Create 3D plot
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
    
    # Plot attractors
ax.scatter(attractor_positions[:, 0], 
            attractor_positions[:, 1], 
            attractor_positions[:, 2], 
            c='blue', alpha=0.6, s=20)
    
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title(f'Attractor Positions - Epoch {0}')
    
plt.tight_layout()
plt.savefig(f'{run}_attractors_epoch_{0}_{'2_layer_potential'}.png', dpi=150)
plt.close()  # Close to free memory
    
print(f'  Attractor positions plotted: shape {attractor_positions.shape}')
print(f'  Attractor range: X=[{attractor_positions[:, 0].min():.2f}, {attractor_positions[:, 0].max():.2f}], '
      f'Y=[{attractor_positions[:, 1].min():.2f}, {attractor_positions[:, 1].max():.2f}], '
      f'Z=[{attractor_positions[:, 2].min():.2f}, {attractor_positions[:, 2].max():.2f}]')   

    


for epoch in range(epochs):
  for images, labels in trainloader:
    optimizer.zero_grad()

    
    # CRITICAL: Move data to GPU
    images = images.to(device)
    labels = labels.to(device)
     

    model_output=model(images)

    loss=loss_function(model_output,labels)

    loss.backward()
    #the optimiser actually takes in the model as an inout so it knows what to update

    # Clip gradients to prevent explosion
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    # Or clip per parameter group
    torch.nn.utils.clip_grad_norm_(encoder_decoder_params, max_norm=1.0)
    torch.nn.utils.clip_grad_norm_(attractor_params, max_norm=0.5)  # Smaller for attractors


    optimizer.step()
    #print("stepped")


  print(f'Epoch [{epoch+1}/{epochs}] Loss: {loss.item():.4f}') 
    # Plot attractor positions at end of each epoch
  with torch.no_grad():
    # Access attractor positions from the first (or all) flap layers
    # Since num_attractor_layers=1, there's one flap_layer

    
    attractor_positions = model.flap_layers[0].attractor_positions_tensor.detach().cpu().numpy()
    # Shape: (num_attractors, 3)
    
    # Create 3D plot
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot attractors
    ax.scatter(attractor_positions[:, 0], 
               attractor_positions[:, 1], 
               attractor_positions[:, 2], 
               c='blue', alpha=0.6, s=20)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(f'Attractor Positions - Epoch {epoch+1}')
    
    plt.tight_layout()
    plt.savefig(f'{run}_attractors_epoch_{epoch+1}_{'2_layer_potential'}.png', dpi=150)
    plt.close()  # Close to free memory
    
    print(f'  Attractor positions plotted: shape {attractor_positions.shape}')
    print(f'  Attractor range: X=[{attractor_positions[:, 0].min():.2f}, {attractor_positions[:, 0].max():.2f}], '
          f'Y=[{attractor_positions[:, 1].min():.2f}, {attractor_positions[:, 1].max():.2f}], '
          f'Z=[{attractor_positions[:, 2].min():.2f}, {attractor_positions[:, 2].max():.2f}]')   

    


correct=0
total=0

with torch.no_grad():
  for images, labels in testloader:
    # CRITICAL: Move data to GPU
    images = images.to(device)
    labels = labels.to(device)

    output=model(images)


    _, predicted=torch.max(output,1)
    total += labels.size(0)
    correct += (predicted==labels).sum().item()

print(f'accuracy: {correct} images out of {total} with {(correct/total)*100} % accuracy')    

#####Ok we are going to try various kinds of regularisation
#####The first one is to apply weight decay to the encoder and decoder parameters

# Extract encoder and decoder weights
encoder_state_dict = model.encoder.state_dict()
decoder_state_dict = model.decoder.state_dict()

print("Encoder weights extracted")
print("Decoder weights extracted")

# Create a baseline model WITHOUT FLAP layers
class BaselineModel(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
    
    def forward(self, x):
        x = x.view(-1, 28*28)
        x = F.gelu(self.encoder(x))
        # NO FLAP LAYERS - direct pass through
        x = F.log_softmax(self.decoder(x), dim=1)
        return x

# Create baseline model with same encoder/decoder architecture
baseline_encoder = nn.Linear(28*28, model.N)
baseline_decoder = nn.Linear(model.N, 10)

# Load the trained weights
baseline_encoder.load_state_dict(encoder_state_dict)
baseline_decoder.load_state_dict(decoder_state_dict)

baseline_model = BaselineModel(baseline_encoder, baseline_decoder)
baseline_model = baseline_model.to(device)

# Test the baseline model (no FLAP)
baseline_correct = 0
baseline_total = 0

with torch.no_grad():
    baseline_model.eval()
    for images, labels in testloader:
        images = images.to(device)
        labels = labels.to(device)
        
        output = baseline_model(images)
        _, predicted = torch.max(output, 1)
        baseline_total += labels.size(0)
        baseline_correct += (predicted == labels).sum().item()

baseline_accuracy = (baseline_correct / baseline_total) * 100
flap_accuracy = (correct / total) * 100

print(f'\n{"="*60}')
print(f'FLAP Model Accuracy: {flap_accuracy:.2f}%')
print(f'Baseline Model Accuracy (No FLAP): {baseline_accuracy:.2f}%')
print(f'Difference: {flap_accuracy - baseline_accuracy:.2f}%')
print(f'{"="*60}')

if abs(flap_accuracy - baseline_accuracy) < 1.0:
    print("WARNING: FLAP layer appears to be doing very little!")
else:
    print("FLAP layer is contributing to the model performance.")

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




class ENC(nn.Module):
    def __init__(self,config): #####Initialises the new params
      super().__init__() ##### Initialises the attributes and methods of the parent
      self.c_fc=nn.Linear(config.d_in,config.d_latent)
      self.gelu=nn.GELU(approximate='tanh')
      self.c_proj=nn.Linear(config.d_latent,config.d_latent)

    def forward(self,x):
      x=self.c_fc(x)
      x=self.gelu(x)
      x= self.c_proj(x)
      return x    



class DEC(nn.Module):
    def __init__(self,config): #####Initialises the new params
      super().__init__() ##### Initialises the attributes and methods of the parent
      self.c_fc=nn.Linear(config.d_latent,config.d_out)
      self.gelu=nn.GELU(approximate='tanh')
      self.c_proj=nn.Linear(config.d_out,config.d_out)

    def forward(self,x):
      x=self.c_fc(x)
      x=self.gelu(x)
      x= self.c_proj(x)
      return x    

      
class potential(nn.Module):
    def __init__(self,config): #####Initialises the new params
      super().__init__() ##### Initialises the attributes and methods of the parent
      self.c_fc=nn.Linear(1,config.potential_hidden_dim)
      self.gelu=nn.GELU(approximate='tanh')
      self.c_proj=nn.Linear(config.potential_hidden_dim,1)

    def forward(self,distances):
      # distances: tensor of shape (...,) containing pairwise distances
      # Reshape to (..., 1) for the linear layer
      x = distances.unsqueeze(-1)
      x = self.c_fc(x)
      x = self.gelu(x)
      x = self.c_proj(x)
      return x.squeeze(-1)  # Return scalar potential values    



class flap_layer(nn.Module):
    def __init__(self,config): #####Initialises the new params
      super().__init__()




@dataclass
class FlapConfig:
    d_in: int = 28*28
    d_latent: int = 3
    d_out: int = 10
    P: int = 1000






















