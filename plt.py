import matplotlib.pyplot as plt
import numpy as np
from collections import OrderedDict
import torch
import torch.nn as nn

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

m = 1.0    # Mass (kg)
c = 0.2    # Damping coefficient (kg/s)
k = 2.0    # Spring constant (N/m)
x0 = 1.0   # Initial displacement (m)
v0 = 0.0   # Initial velocity (m/s)
# Time domain
t_min = 0.0
t_max = 10.0

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

    def forward(self, x):
        out = self.layers(x)
        return out

model = PINN(
    input_size=1,
    hidden_size=20,
    output_size=1,
    depth=4,
    act=torch.nn.Tanh
)

model.load_state_dict(torch.load('models/model.ckpt', map_location=device, weights_only=True))

model.eval()

# Calculate analytical solution
omega_n = np.sqrt(k / m)
zeta = c / (2 * np.sqrt(k * m))

if zeta < 1:
   # Under-damped case
   omega_d = omega_n * np.sqrt(1 - zeta**2)
   A = x0
   B = (v0 + zeta * omega_n * x0) / omega_d
   t_analytical = np.linspace(t_min, t_max, 1000)
   x_analytical = np.exp(-zeta * omega_n * t_analytical) * (
         A * np.cos(omega_d * t_analytical) +
         B * np.sin(omega_d * t_analytical)
         )
else:
   # Critically damped or over-damped
   t_analytical = np.linspace(t_min, t_max, 1000)
   x_analytical = np.zeros_like(t_analytical)

# Predict displacement using the trained model
t_test = t_analytical.reshape(-1, 1)
x_pinn = model(torch.tensor(t_test, dtype=torch.float32))
x_pinn = x_pinn.detach().numpy().flatten()

# Plot the results
plt.figure(figsize=(10, 6))
plt.plot(t_analytical, x_analytical, label='Analytical Solution', linestyle='dashed')
plt.plot(t_test, x_pinn, label='PINN Solution')
plt.xlabel('Time (s)')
plt.ylabel('Displacement (m)')
plt.title('Spring-Mass-Damper System Response')
plt.legend()
plt.grid(True)
plt.show()