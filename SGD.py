# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import LiqFunDobPen as lf
import ParametersLiq as lp

# %%
df = pd.read_csv("double_pendulum_dataset.csv") # Charge the csv(DATA)
# We have to turn it into an array
data = df.values
# Separate inputs and outputs
X = data[:, :4] # Inputs
Y = data[:, 4:] # Outputs

# %%
# Computation of the mean and std 
X_mean, X_std = lf.compute_stats(X)
Y_mean, Y_std = lf.compute_stats(Y)
# Normalization
X_norm = lf.norm(X, X_mean, X_std)
Y_norm = lf.norm(Y, Y_mean, Y_std)

# %%
# We need time sequences
X_seq = []
Y_seq = []
for i in range(len(X)-lp.seq_len):
    X_seq.append(X_norm[i:i+lp.seq_len])
    Y_seq.append(Y_norm[i:i+lp.seq_len])
X_seq = np.array(X_seq)
Y_seq = np.array(Y_seq)

# %%
net = lf.LiquidNetwork(
    input_size=X.shape[1],
    hidden_size=lp.hidden_size,
    output_size=Y.shape[1]
)

for epoch in range(lp.epochs):

    total_loss = 0
    lr = lp.initial_lr * (0.95 ** epoch)

    for i in range(len(X_seq)):

        net.reset()

        states = []
        outputs = []
        inputs = []
        targets = []

        loss = 0

        # ===== FORWARD =====
        for t in range(len(X_seq[i])):

            x_t = X_seq[i, t]
            y_t = Y_seq[i, t]

            y_pred = net.step(x_t)

            states.append(net.state.copy())
            outputs.append(y_pred)
            inputs.append(x_t)
            targets.append(y_t)

            loss += lf.mse_loss(y_pred, y_t)
            loss = loss / len(X_seq[i])

        # ===== BACKWARD (BPTT) =====
        grad_W_in = np.zeros_like(net.W_in)
        grad_W_rec = np.zeros_like(net.W_rec)
        grad_W_out = np.zeros_like(net.W_out)
        grad_bias = np.zeros_like(net.bias)
        grad_b_out = np.zeros_like(net.bias_out)

        grad_state_next = np.zeros(net.hidden_size)

        for t in reversed(range(len(states))):

            state_t = states[t]
            input_t = inputs[t]
            y_pred = outputs[t]
            y_true = targets[t]

            # grad loss
            grad_y = lf.mse_grad(y_pred, y_true)

            # output layer
            grad_W_out += np.outer(grad_y, state_t)
            grad_b_out += grad_y

            grad_state = net.W_out.T @ grad_y
            grad_state += grad_state_next

            # recomputar z
            z = net.W_in @ input_t + net.W_rec @ state_t + net.bias
            tanh_z = np.tanh(z)
            dtanh = 1 - tanh_z**2

            grad = dtanh * grad_state

            # gradientes
            grad_W_rec += np.outer(grad, state_t)
            grad_W_in += np.outer(grad, input_t)
            grad_bias += grad

            # BPTT
            grad_state_next = net.W_rec.T @ grad

        # ===== CLIPPING =====
        grad_W_rec = np.clip(grad_W_rec, -0.7, 0.7)
        grad_W_in = np.clip(grad_W_in, -0.7, 0.7)

        
        grad_W_out /= len(states)
        grad_b_out /= len(states)

        # ===== UPDATE =====
        net.W_in  -= lr * grad_W_in
        net.W_rec -= lr * grad_W_rec
        net.W_out -= lr * grad_W_out

        net.bias     -= lr * grad_bias
        net.bias_out -= lr * grad_b_out


        total_loss += loss

    print(f"Epoch {epoch}, Loss: {total_loss}")

# %%



