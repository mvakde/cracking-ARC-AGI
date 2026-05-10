import torch
import torch.nn.functional as F
torch.manual_seed(0)

# Tiny network: single linear layer (so we can see logits clearly)
in_dim, num_classes = 4, 3
net = torch.nn.Linear(in_dim, num_classes, bias=True)

# Random single example
x = torch.randn(1, in_dim)
y = torch.tensor([2])  # choose some target class (0..num_classes-1)
y_onehot = F.one_hot(y, num_classes=num_classes).float()
print("y_onehot:", y_onehot.tolist())

lr = 0.1
opt = torch.optim.SGD(net.parameters(), lr=lr)

# Forward, loss, backward
opt.zero_grad()
logits_before = net(x)
logits_before.retain_grad()  # so we can see dL/d(logits)
print("x:", x.tolist())
print("logits BEFORE:", logits_before.detach().tolist())

p = logits_before.softmax(dim=-1)
log_probs = logits_before.log_softmax(dim=-1)
loss = -(y_onehot * log_probs).sum(dim=-1).mean()
loss.backward()

# Gradients w.r.t. logits are exactly p - y (this is the key identity)
print("softmax probs p:", p.detach().tolist())
print("grad wrt logits (should be p - y):", logits_before.grad.detach().tolist())

# One SGD step
with torch.no_grad():
    opt.step()

# Forward again on the SAME input
logits_after = net(x)
print("logits AFTER: ", logits_after.detach().tolist())

# ===== Analytic check for a single linear layer =====
# For z = x W^T + b, SGD on W,b with step lr yields:
# z_after = z_before - lr * [ (x (dL/dW)^T) + (dL/db) ]
# With dL/dW = (p - y) ⊗ x  and dL/db = (p - y),
# x (dL/dW)^T = (||x||^2) * (p - y)
# => z_after = z_before - lr * (||x||^2 + 1) * (p - y)
g = p - y_onehot
scale = (x.pow(2).sum() + 1.0)
predicted_after = logits_before.detach() - lr * scale * g.detach()

print("||x||^2 + 1 =", scale.item())
print("Predicted logits AFTER (closed-form):", predicted_after.tolist())

# Verify equality up to numerical precision
max_abs_err = (logits_after.detach() - predicted_after).abs().max().item()
print("Max |actual - predicted|:", max_abs_err)
assert torch.allclose(logits_after.detach(), predicted_after, atol=1e-6), "Mismatch!"

# Show proportionality for non-targets: Δz_j / (-p_j) is constant = lr*(||x||^2+1)
delta = (logits_after - logits_before).detach()[0]     # shape [C]
p_vals = p.detach()[0]                                  # shape [C]
t = y.item()

const = (lr * (x.pow(2).sum() + 1.0)).item()
mask = torch.ones_like(p_vals, dtype=torch.bool)
mask[t] = False

ratios = (delta[mask] / (-(p_vals[mask].clamp_min(1e-12)))).tolist()
print("Δlogit for non-targets:", [float(delta[j]) for j in range(num_classes) if j != t])
print("p_j for non-targets    :", [float(p_vals[j]) for j in range(num_classes) if j != t])
print("Δlogit_j / (-p_j):      ", ratios)
print("Expected constant       =", const)
