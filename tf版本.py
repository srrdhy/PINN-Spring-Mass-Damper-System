import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

tf.random.set_seed(42)
np.random.seed(42)

m = 1.0    # Mass (kg)
c = 0.2    # Damping coefficient (kg/s)
k = 2.0    # Spring constant (N/m)

# External force function F(t)
def F(t):
   return 0.0  # No external force (free vibration)

# Initial conditions
x0 = 1.0   # Initial displacement (m)
v0 = 0.0   # Initial velocity (m/s)

# Time domain
t_min = 0.0
t_max = 10.0

# Number of collocation points
N_f = 1000  # For the ODE residual
N_i = 1     # For the initial conditions

# Collocation points in the interior of the domain
t_f = np.linspace(t_min, t_max, N_f).reshape(-1, 1)

# Initial condition point
t_i = np.array([[t_min]])

# Define the neural network using the Keras API
class PINN(tf.keras.Model):
   def __init__(self):
       super(PINN, self).__init__()
       self.hidden_layers = []
       num_hidden_layers = 4
       num_neurons_per_layer = 20

       # Initialize hidden layers
       for _ in range(num_hidden_layers):
           layer = tf.keras.layers.Dense(
               num_neurons_per_layer,
               activation='tanh',
               kernel_initializer='glorot_normal'
           )
           self.hidden_layers.append(layer)

       # Output layer for x(t)
       self.out_layer = tf.keras.layers.Dense(1, activation=None)

   def call(self, t):
       X = t
       for layer in self.hidden_layers:
           X = layer(X)
       x = self.out_layer(X)
       return x
   
# Define the optimizer
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)

# Define the loss function
def loss_fn(model, t_f, t_i, x0, v0):
   # Compute the ODE residuals at collocation points
   with tf.GradientTape(persistent=True) as tape:
       tape.watch(t_f)
       x = model(t_f)
       x_t = tape.gradient(x, t_f)
   x_tt = tape.gradient(x_t, t_f)

   # ODE residual
   f_ode = m * x_tt + c * x_t + k * x - F(t_f)

   # Mean squared residual of ODE
   f_ode_loss = tf.reduce_mean(tf.square(f_ode))

   # Initial conditions
   x_i = model(t_i)
   with tf.GradientTape() as tape_ic:
       tape_ic.watch(t_i)
       x_i = model(t_i)
   x_t_i = tape_ic.gradient(x_i, t_i)

   ic_loss = tf.square(x_i - x0) + tf.square(x_t_i - v0)

   total_loss = f_ode_loss + ic_loss

   return total_loss

# Convert data to tensors
t_f_tensor = tf.convert_to_tensor(t_f, dtype=tf.float32)
t_i_tensor = tf.convert_to_tensor(t_i, dtype=tf.float32)
x0_tensor = tf.convert_to_tensor([[x0]], dtype=tf.float32)
v0_tensor = tf.convert_to_tensor([[v0]], dtype=tf.float32)

# Initialize the model
model = PINN()

# Define the optimizer 
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3) # Moved the optimizer definition here

# Training loop
epochs = 5000
for epoch in range(epochs):
   with tf.GradientTape() as tape:
       loss_value = loss_fn(model, t_f_tensor, t_i_tensor, x0_tensor, v0_tensor)
   grads = tape.gradient(loss_value, model.trainable_variables)
   optimizer.apply_gradients(zip(grads, model.trainable_variables))

   if epoch % 500 == 0:
       print(f"Epoch {epoch}, Loss: {loss_value.numpy().item():.5f}")