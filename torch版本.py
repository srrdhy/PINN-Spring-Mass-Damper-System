import torch
import torch.nn as nn
import numpy as np
import os
import math
from collections import OrderedDict

if not os.path.isdir('./models'):
        os.mkdir('./models')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

def same_seed(seed): 

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

same_seed(42)

m = 1.0    # Mass (kg)
c = 0.2    # Damping coefficient (kg/s)
k = 2.0    # Spring constant (N/m)

# External force function F(t)
def F(t):
   return 0.0  # No external force (free vibration)

# Initial conditions
x0 = 1.0   # Initial displacement (m)
v0 = 0.0   # Initial velocity (m/s)
#x0 = torch.tensor([[x0]], dtype=torch.float32, device=device)
#v0 = torch.tensor([[v0]], dtype=torch.float32, device=device)

# Time domain
t_min = 0.0
t_max = 10.0

# Number of collocation points
N_f = 1000  # For the ODE residual
N_i = 1     # For the initial conditions

# Collocation points in the interior of the domain
t_f = np.linspace(t_min, t_max, N_f).reshape(-1, 1)
t_f = torch.tensor(t_f, dtype=torch.float32, requires_grad=True).to(device)

# Initial condition point
t_i = np.array([[t_min]])
t_i = torch.tensor(t_i, dtype=torch.float32, requires_grad=True).to(device)

# Define the neural network
class PINN(nn.Module):  
    def __init__(
        self,
        input_size,
        hidden_size,
        output_size,
        depth,
        act=torch.nn.Tanh,
    ):
        super(PINN, self).__init__()
        
        layers = [('input', torch.nn.Linear(input_size, hidden_size))] # Input

        layers.append(('input_activation', act())) # Activation

        for i in range(depth): 
            layers.append(('hidden_%d' % i, torch.nn.Linear(hidden_size, hidden_size))) # Hidden Layer
            layers.append(('activation_%d' % i, act())) # Activation for Hidden Layer

        layers.append(('output', torch.nn.Linear(hidden_size, output_size)))

        layerDict = OrderedDict(layers)
        self.layers = torch.nn.Sequential(layerDict)

    # 添加权重初始化
        for name, param in self.layers.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)

    def forward(self, x):
        out = self.layers(x)
        return out

# 计算导数
def calculate_grad(y, x):
        return torch.autograd.grad(y, x, torch.ones_like(y), retain_graph=True, create_graph=True)[0]


def pde_loss(model, t_f):

    x = model(t_f)
    
    x_t = calculate_grad(x, t_f)
    x_tt = calculate_grad(x_t, t_f)

    pde = m * x_tt + c * x_t + k * x - F(t_f)
    loss = torch.mean(pde**2)

    return loss

def ini_loss(model, t_i):

    x_i = model(t_i)

    x_i_t = calculate_grad(x_i, t_i)

    pde1 = x_i_t - v0
    pde2 = x_i - x0

    loss = torch.mean(pde1**2) + torch.mean(pde2**2)

    return loss

def train(model):
    # Define the optimizer
    optimizer_adam = torch.optim.Adam(model.parameters(), lr=1e-3)

    n_epochs, best_loss, early_stop_count = 5000, math.inf, 0

    for epoch in range(n_epochs):
        optimizer_adam.zero_grad()

        loss_pde = pde_loss(model, t_f)
        loss_bd_ic = ini_loss(model, t_i)
        loss = loss_pde + loss_bd_ic
        loss.backward()
        optimizer_adam.step()

        if (epoch) % 500 == 0:
            print(f'Epoch [{epoch}/{n_epochs}]: Train loss: {loss:.5f}') 
            print('Saving model with loss {:.5f}...'.format(best_loss))
            
        if loss < best_loss:
            best_loss = loss
            torch.save(model.state_dict(), './models/model.ckpt') # Save your best model
            early_stop_count = 0
        else: 
            early_stop_count += 1

        if early_stop_count >= 500: 
            print('\nModel is not improving, so we halt the training session.')
            break

model = PINN(
            input_size=1,
            hidden_size=20,
            output_size=1,
            depth=4,
            act=torch.nn.Tanh
        ).to(device)
train(model) 