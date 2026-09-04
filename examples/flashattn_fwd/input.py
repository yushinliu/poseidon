def torch_fn(q, k, v):
    scale = q.shape[-1] ** -0.5
    s = (q @ k.transpose(-2, -1)) * scale
    p = torch.softmax(s, dim=-1)
    return p @ v
