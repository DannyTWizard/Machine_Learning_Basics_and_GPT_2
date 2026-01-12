#####We are going to load the huggingface weights into a custom class


import torch
import torch.nn as nn
from torch.nn import functional as F

import matplotlib.pyplot as plt
import transformers

from transformers import GPT2Tokenizer, GPT2LMHeadModel

import numpy as np
import math



#####Then we are going to intitialise the model randomly and train it ourselves

import dataclasses
from dataclasses import dataclass

print("running")

#####The multi headed attention in the og tensorflow implementation runs many attention heads in parallel and concatenates the results
#### with a module list of heads

####This time we will use a single self attention module with a bunch of tensor manips

class CausalSelfAttention(nn.Module):
  def __init__(self,config):
    super().__init__()
    assert config.n_embd % config.n_head==0 ####This makes sure you can divide each embedding vector among the heads
    ###This layer will have some weights that will generate a a key, query and value vector for each vector you pass through it
    ###Eact of the q, k, v vectors will have a dimension of  however big the head is
    self.c_attn=nn.Linear(config.n_embd,3*config.n_embd)
    ####This layer will project the out put ig
    self.c_proj=nn.Linear(config.n_embd,config.n_embd)
    self.n_head=config.n_head
    self.n_embd=config.n_embd
    self.register_buffer("bias", torch.tril(torch.ones(config.block_size,config.block_size)).view(1,1,config.block_size,config.block_size))

  def forward(self,x):

    B,T,C=x.size() #C is nh * hs

    qkv=self.c_attn(x)

    q,k,v =qkv.split(self.n_embd, dim =2)
    k=k.view(B,T, self.n_head, C//self.n_head).transpose(1,2)
    q=q.view(B,T, self.n_head, C//self.n_head).transpose(1,2)
    v=v.view(B,T, self.n_head, C//self.n_head).transpose(1,2)


    ####Make the attention pattern and normalise it
    att=(q @ k.transpose(-2,-1) * (1.0/math.sqrt(k.size(-1))))
    ####Mask the retrospective attention scores
    att=att.masked_fill(self.bias[:,:,:T,:T]==0,float('-inf'))

    ####Put a soft max on the attention columns
    att=F.softmax(att, dim=-1)
    ####multiplyby the value vectors
    y=att @ v


    y=y.transpose(1,2).contiguous().view(B,T,C)
    #output projection
    y=self.c_proj(y)

    return y #####The delta vs you will add to the lookup embeddings



class MLP(nn.Module):
    def __init__(self,config): #####Initialises the new params
      super().__init__() ##### Initialises the attributes and methods of the parent
      self.c_fc=nn.Linear(config.n_embd,4*config.n_embd)
      self.gelu=nn.GELU(approximate='tanh')
      self.c_proj=nn.Linear(4*config.n_embd,config.n_embd)

    def forward(self,x):
      x=self.c_fc(x)
      x=self.gelu(x)
      x= self.c_proj(x)
      return x




class Block(nn.Module):

  def __init__(self,config): #####Initialises the new params
    super().__init__() ##### Initialises the attributes and methods of the parent

    self.ln_1=nn.LayerNorm(config.n_embd)
    self.attn=CausalSelfAttention(config)
    self.ln_2=nn.LayerNorm(config.n_embd)
    self.mlp=MLP(config)


  def forward(self,x):
    x=x+self.attn(self.ln_1(x))
    x=x+self.mlp(self.ln_2(x))
    return x









@dataclass
class GPTconfig:
  block_size: int = 1024
  vocab_size: int = 50257
  n_layer: int = 12
  n_head: int =12
  n_embd: int =768



class GPT(nn.Module):

  def __init__(self,config):
    super().__init__()
    self.config=config
    #####GPT2 is a decoder only transformer

    self.transformer=nn.ModuleDict(dict(
        wte=nn.Embedding(config.vocab_size,config.n_embd),
        ####nn embedding is a tensor wrapper that lets u index the rows
        ####just a set of vectors
        wpe=nn.Embedding(config.block_size,config.n_embd),
        h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
        ln_f=nn.LayerNorm(config.n_embd),
    ))
    self.lm_head=nn.Linear(config.n_embd,config.vocab_size, bias=False)

  def forward(self,idx):
    #idx is an array-like object
    B,T= idx.size()
    assert T<= self.config.block_size, f"Cannot forward sequence of length {T}, block size is only {self.config.block_size}"
    pos=torch.arange(0,T, dtype=torch.long, device=idx.device)
    pos_emb=self.transformer.wpe(pos)
    tok_emb=self.transformer.wte(idx)
    x = tok_emb +pos_emb
    #pass x forward through the blocks
    for block in self.transformer.h:
      x=block(x)
    #apply the final layer norm and get the logits
    x=self.transformer.ln_f(x)
    logits=self.lm_head(x)
    return logits  






  @classmethod
  def from_pretrained(cls,model_type):
    """Loads pre trained gpt 2 weights from hugging face"""
    assert model_type in {'gpt2','gpt2-medium','gpt2-large','gpt2-xl'}
    from transformers import GPT2LMHeadModel
    print('loading weights from pretrained gpt: %s' % model_type)
    #n_layers, n_heads, n_embd, are determined from the model type
    config_args={
      'gpt2':dict(n_layer=12,n_head=12,n_embd=768),
      'gpt2-medium':dict(n_layer=24,n_head=16,n_embd=1024),
      'gpt2-large':dict(n_layer=36,n_head=20,n_embd=1280),
      'gpt2-xl':dict(n_layer=48,n_head=24,n_embd=1600)


    }[model_type]

    config_args['vocab_size']=50257
    config_args['block_size']=1024

        ###Now create from scratch aninitialised min gpt model

    config=GPTconfig(**config_args)
    model=GPT(config)

    sd=model.state_dict()
    sd_keys=sd.keys()

        #we ignore the parameters that are the masks used in learning things like the attention weights
    sd_keys=[k for k in sd_keys if not k.endswith('.attn.bias')]


        #init a huggingface transformers model

    model_hf=GPT2LMHeadModel.from_pretrained(model_type)
    sd_hf=model_hf.state_dict()
    sd_keys_hf=sd_hf.keys()
    sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')] # ignore these, just a buffer
    sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')] # same, just the mask (buffer)
    transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them
    assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
    for k in sd_keys_hf:
      if any(k.endswith(w) for w in transposed):
            #special treatment for the conv1D weights we wanna transpose
        assert sd_hf[k].shape[::-1] == sd[k].shape
        with torch.no_grad():
          sd[k].copy_(sd_hf[k].t())
      else:
            #just copy over the other params without transposing
        assert sd_hf[k].shape == sd[k].shape
        with torch.no_grad():
          sd[k].copy_(sd_hf[k])

    return model













    #####In the og transformer paper you had layer norms inside the residual stream
    #####This is not good because you want a clean residual stream.



model=GPT.from_pretrained('gpt2')
print('were not cooked UwU')

num_return_sequences=5
max_length=60
model=GPT.from_pretrained('gpt2')
model.eval()


import os, torch
print('CUDA_VISIBLE_DEVICES =', os.environ.get('CUDA_VISIBLE_DEVICES'))
print('device_count        =', torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    print(i, torch.cuda.get_device_name(i))
print('current_device      =', torch.cuda.current_device())

device = torch.device('cuda:0')  # or 'cuda:1'
# optional: make it the process default
#torch.cuda.set_device(device)

print("device",device)

########The entire model is run on a GPU
model.to(device)
#device = next(model.parameters()).device
#print(torch.cuda.is_available(), torch.cuda.device_count())
#print(next(model.parameters()).device)
#print(next(model.parameters()).is_cuda)
#print(model.is_cuda, model.device)



##### get an tensor of tokens from tiktokeniser

import tiktoken

enc=tiktoken.get_encoding('gpt2')
tokens=enc.encode("Hello, I'm a language model,")
tokens=torch.tensor(tokens, dtype=torch.long)
tokens=tokens.unsqueeze(0).repeat(num_return_sequences,1)
#x=tokens.to('cuda')
x=tokens.to(device)
print("xloc",x.is_cuda, x.device) 
#print(model.is_cuda, model.device)
while x.size(1)<max_length:
  print('cuda fine')
  with torch.no_grad():
    logits=model(x)
    ####Take the logits at the last position
    logits= logits[:,-1,:]
    #get the probabilities
    probs=F.softmax(logits,dim=-1)
    #do topk sampling of 50 (hf default)
    topk_probs, topk_indices =torch.topk(probs, 50, dim=-1)
    #select a token from the topk probabilities
    # Add a small epsilon to the probabilities to avoid issues with near-zero values
    print("probs",topk_probs)
    topk_probs=torch.where(topk_probs<0,0,topk_probs)
    topk_probs=torch.where(torch.isnan(topk_probs),1,topk_probs)
    #topk_probs=torch.where(torch.isinf(topk_probs),1,topk_probs)
    ix=torch.multinomial(topk_probs,1)
    #gather the corresponding indices
    xcol=torch.gather(topk_indices,-1,ix)
    #append to the sequence
    x=torch.cat((x,xcol),dim=1)



#print the results
for i in range (num_return_sequences):
  tokens=x[i, :max_length].tolist()
  decoded=enc.decode(tokens)
  print(">",decoded)



