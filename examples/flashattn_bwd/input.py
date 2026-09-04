def torch_fn(q, k, v, do):
    # FlashAttention 反向参考实现（无因果掩码、无 dropout）
    scale = q.shape[-1] ** -0.5
    s = (q @ k.transpose(-2, -1)) * scale
    p = torch.softmax(s, dim=-1)
    dp = do @ v.transpose(-2, -1)
    dv = p.transpose(-2, -1) @ do
    ds = p * (dp - (dp * p).sum(dim=-1, keepdim=True))
    dq = (ds * scale) @ k
    dk = (ds * scale).transpose(-2, -1) @ q
    return dq, dk, dv
