import numpy as np
import pandas as pd
import torch
import torch.nn as nn 

# LIQUID NEURON
class LiquidNeuron:
    def __init__(self, input_size):
        self.input_size = input_size

    def activation(self, x):
        return np.tanh(x) #Activation function

# LIQUID NETWORK
class LiquidNetwork:
    def __init__(self, input_size, hidden_size, output_size):

        self.hidden_size = hidden_size # Save  dynamic size
        self.input_size = input_size
        self.output_size = output_size

        self.neuron = LiquidNeuron(input_size) # Auxiliar neuron
        self.tau = np.random.uniform(0.85, 2.0, size=hidden_size) # Time constants; (tau1, tau2,..., taun)

        # Input weights
        self.W_in = np.random.randn(hidden_size, input_size) * 0.1
        # Recurrent weights
        self.W_rec = np.random.randn(hidden_size, hidden_size) * 0.1
        #Bias, displacement
        self.bias = np.zeros(hidden_size)
        # x(t)
        self.state = np.zeros(hidden_size)
        # Output; Turns state into an output
        self.W_out = np.random.randn(output_size, hidden_size) * 0.1
        # Output bias
        self.bias_out = np.zeros(output_size)

    def step(self, input_signal, dt=0.01): # Computation of the dynamics

        dx = (-self.state + self.neuron.activation(
            self.W_in @ input_signal +
            self.W_rec @ self.state +             # Final dynamics, dx/dt= -x + tanh(...)
            self.bias
        )) / (self.tau + 1e-6)
        

        self.state += dt * dx  # Euler

        return self.W_out @ self.state + self.bias_out # Linear projection; y = Wout*x + b
    
    def reset(self):
        self.state = np.zeros(self.hidden_size)


# COMPUTATION OF THE MEAN AND STD
def compute_stats(x):
    X_mean = x.mean(axis=0)
    X_std = x.std(axis=0) + 1e-6
    return X_mean, X_std

#NORMALIZATION
def norm(x, x_mean, x_std):
    return (x - x_mean) / x_std

# LOSS
def mse_loss(y_pred, y_true):
    return np.mean((y_pred-y_true)**2)

# LOSS GRADIENT
def mse_grad(y_pred, y_true):
    return 2 * (y_pred - y_true)/ y_pred.size 

#SIMULATION
def simulate(net, x0, steps, dt=0.01):

    net.reset()

    x = x0.copy()

    trajectory = [x.copy()]

    for _ in range(steps):

        # predicción de aceleraciones
        alpha = net.step(x)

        # separar variables
        theta = x[:2]
        omega = x[2:]

        # integrar
        omega = omega + dt * alpha
        theta = theta + dt * omega

        # nuevo estado
        x = np.concatenate([theta, omega])

        trajectory.append(x.copy())

    return np.array(trajectory)