import numpy as np
import pandas as pd
import torch
import torch.nn as nn 
import ParametersLiq as lp

# LIQUID NEURON
class LiquidNeuron:
    def __init__(self, input_size):
        self.input_size = input_size

    def activation(self, x):
        return np.tanh(x) #Activation function

# LIQUID NETWORK
class LiquidNetwork:

    def __init__(self, input_size, hidden_size, output_size):

        self.hidden_size = hidden_size
        self.input_size = input_size
        self.output_size = output_size

        # Pesos principales
        self.W_in = np.random.randn(hidden_size, input_size) * 0.1
        
        # Recurrente ortogonal 🔥
        Q, _ = np.linalg.qr(np.random.randn(hidden_size, hidden_size))
        self.W_rec = 0.95 * Q

        self.bias = np.zeros(hidden_size)

        # Estado
        self.state = np.zeros(hidden_size)

        # Output
        self.W_out = np.random.randn(output_size, hidden_size) * 0.1
        self.bias_out = np.zeros(output_size)

        # Tau dinámico ✅
        self.W_tau_in = np.random.randn(hidden_size, input_size) * 0.05
        self.W_tau_rec = np.random.randn(hidden_size, hidden_size) * 0.02
        self.b_tau = np.zeros(hidden_size)
        self.W_skip = np.random.randn(output_size, input_size) * 0.1


    def activation(self, z):
        return np.tanh(z) + 0.2 * np.sin(z)

    def step(self, input_signal, dt=0.01):

        # Tau(constant) but changes over time(liquid)
        tau = np.exp(
            self.W_tau_in @ input_signal +
            self.W_tau_rec @ self.state +
            self.b_tau
        )
        tau = np.clip(tau, 0.1, 2.0)

        z = self.W_in @ input_signal + self.W_rec @ self.state + self.bias

        dh_dt = (-self.state + self.activation(z)) / (tau + 1e-6)

        # estabilidad
        dh_dt = np.clip(dh_dt, -5, 5)

        self.state += dt * dh_dt
        self.state = np.clip(self.state, -2.0, 2.0)

        # skip connection ✅
        return (self.W_out @ self.state + self.bias_out + self.W_skip @ input_signal) * 0.5

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

def rk4_step(net, x, dt):

    state_backup = net.state.copy()

    def f(x_local):
        return np.concatenate([
            x_local[2:],           # omega
            net.step(x_local)      # alpha
        ])

    k1 = f(x)
    net.state = state_backup.copy()

    k2 = f(x + 0.5 * dt * k1)
    net.state = state_backup.copy()

    k3 = f(x + 0.5 * dt * k2)
    net.state = state_backup.copy()

    k4 = f(x + dt * k3)
    net.state = state_backup.copy()

    x_next = x + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

    return x_next

def adam_update(param, grad, m, v, t_adam, lr):
    m = lp.beta1 * m + (1 - lp.beta1) * grad
    v = lp.beta2 * v + (1 - lp.beta2) * (grad**2)
    m_hat = m / (1 - lp.beta1**t_adam)
    v_hat = v / (1 - lp.beta2**t_adam)
    param -= lr * m_hat / (np.sqrt(v_hat) + lp.eps)
    return m, v

def energy(x):
    theta1, theta2, omega1, omega2 = x

    # simplificado (vale para empezar)
    return 0.5 * (omega1**2 + omega2**2) + \
           0.5 * (theta1**2 + theta2**2)

def real_energy(x):
    theta1, theta2, omega1, omega2 = x
    
    return (
        0.5 * (omega1**2 + omega2**2)
        + (1 - np.cos(theta1))
        + (1 - np.cos(theta2))
    )