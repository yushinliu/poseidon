def torch_fn(x, weight, bias):
    return torch.nn.functional.layer_norm(x, x.shape[-1:], weight, bias, eps=1e-5)
