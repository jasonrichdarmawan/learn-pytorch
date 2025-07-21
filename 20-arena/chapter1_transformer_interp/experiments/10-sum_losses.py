# %%

import torch as t

# Setup
mode = 2
x = t.arange(
  0, 
  mode * 2, 
  dtype=t.float32, 
  requires_grad=True
).reshape(mode, 2)
x_orig = x.clone().detach()

# Define two losses: each only depends on one slice
def loss_even(x):  # Only x[0]
    return (x[0] ** 2).sum()

def loss_odd(x):   # Only x[1]
    return ((x[1] - 1) ** 2).sum()

# --- Approach 1: Backprop each loss separately ---
x1 = x_orig.clone().detach().requires_grad_()
opt1 = t.optim.SGD([x1], lr=0.1)

# Step 1: loss_even
opt1.zero_grad()
le = loss_even(x1)
le.backward()
opt1.step()

# Step 2: loss_odd (now x1[0] has changed!)
opt1.zero_grad()
lo = loss_odd(x1)
lo.backward()
opt1.step()

x1_result = x1.detach().clone()

# --- Approach 2: Sum losses and backprop once ---
x2 = x_orig.clone().detach().requires_grad_()
opt2 = t.optim.SGD([x2], lr=0.1)

opt2.zero_grad()
le2 = loss_even(x2)
lo2 = loss_odd(x2)
loss_total = le2 + lo2
loss_total.backward()
opt2.step()

x2_result = x2.detach().clone()

print("Original x:\n", x_orig)
print("After sequential updates:\n", x1_result)
print("After summed update:\n", x2_result)