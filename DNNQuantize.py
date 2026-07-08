import torch
import torch.nn as nn
import quantizer

try:
    from torch.func import functional_call
except Exception:  # pragma: no cover - compatibility for older PyTorch versions
    from torch.nn.utils.stateless import functional_call


def flat_vector_to_param_buffer_dict(model, flat_vector):
    """Build a differentiable parameter/buffer dictionary from a flat vector."""
    params_and_buffers = {}

    for name, param in model.named_parameters():
        params_and_buffers[name] = param

    for name, buffer in model.named_buffers():
        params_and_buffers[name] = buffer

    offset = 0
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        numel = param.numel()
        piece = flat_vector[offset:offset + numel]
        params_and_buffers[name] = piece.view_as(param).to(
            device=param.device,
            dtype=param.dtype,
        )
        offset += numel

    if offset != flat_vector.numel():
        raise ValueError(
            "Flat vector size mismatch: used {}, but vector has {}".format(
                offset,
                flat_vector.numel(),
            )
        )

    return params_and_buffers


def differentiable_task_loss_from_flat_vector(
    model,
    flat_vector,
    train_loader,
    criterion,
    device,
    max_batches=1,
):
    """Evaluate task loss after replacing model parameters by ``flat_vector``."""
    model.eval()
    params_and_buffers = flat_vector_to_param_buffer_dict(model, flat_vector)

    total_loss = torch.tensor(0.0, device=device)
    total_samples = 0
    correct = 0

    for batch_idx, (data, label) in enumerate(train_loader):
        data = data.to(device, non_blocking=True)
        label = label.to(device, non_blocking=True)

        output = functional_call(model, params_and_buffers, (data,))
        total_loss = total_loss + criterion(output, label)
        total_samples += label.numel()

        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(label.view_as(pred)).sum().item()

        if max_batches is not None and max_batches > 0 and batch_idx + 1 >= max_batches:
            break

    total_loss = total_loss / max(total_samples, 1)
    accuracy = 100.0 * correct / max(total_samples, 1)

    return accuracy, total_loss


class PyTorchMLP(torch.nn.Module):
    """Auxiliary network whose output is reshaped into a generator matrix."""

    def __init__(self, num_hidden1=700, num_hidden2=900, num_hidden3=700, output_dim=2):
        super().__init__()
        self.output_dim = output_dim
        self.layer1 = torch.nn.Linear(200, num_hidden1)
        self.layer2 = torch.nn.Linear(num_hidden1, num_hidden2)
        self.layer3 = torch.nn.Linear(num_hidden2, num_hidden3)
        self.layer4 = torch.nn.Linear(num_hidden3, 800)
        self.layer5 = torch.nn.Linear(800, output_dim ** 2)
        self.relu = nn.LeakyReLU()
        self.output_activation = torch.tanh

    def forward(self, inp):
        inp = inp.reshape([-1, 200])
        x = self.relu(self.layer1(inp))
        x = self.relu(self.layer2(x))
        x = self.relu(self.layer3(x))
        x = self.relu(self.layer4(x))
        x = self.output_activation(self.layer5(x))
        return torch.reshape(x, [self.output_dim, self.output_dim])


def train_model(
    model,
    args,
    criterion,
    testData=None,
    optimizer=None,
    train_loader=None,
    anotherModel=None,
):
    """Train the auxiliary network that outputs the lattice generator matrix."""
    if optimizer is None:
        raise ValueError("optimizer must be provided")
    if testData is None:
        return None
    if torch.all(testData.eq(0)):
        return None

    model.train()
    device = next(model.parameters()).device
    local_weights_orig = testData.to(device)

    gen_mat = model(torch.ones(200, device=device))
    mechanism = quantizer.LatticeQuantization(args, gen_mat, True)

    reconstructed_points, vec = mechanism(
        input=local_weights_orig,
        shouldPrint=False,
        shouldReturnBack=True,
    )

    if args.loss_by == "accuracy":
        if anotherModel is None:
            raise ValueError("loss_by='accuracy' requires anotherModel.")

        task_criterion = torch.nn.CrossEntropyLoss(reduction="sum")
        _, loss = differentiable_task_loss_from_flat_vector(
            model=anotherModel,
            flat_vector=reconstructed_points,
            train_loader=train_loader,
            criterion=task_criterion,
            device=args.device,
            max_batches=getattr(args, "task_loss_batches", 1),
        )

    elif args.loss_by == "snr":
        signal_power = torch.var(vec, unbiased=False)
        noise_power = torch.var(vec - reconstructed_points, unbiased=False)
        loss = -10.0 * torch.log10(signal_power / (noise_power + 1e-12))

    else:
        loss = criterion(reconstructed_points, vec)

    optimizer.zero_grad()
    loss.backward()

    if getattr(args, "dnn_quantize_verbose", False):
        grad_norm = 0.0
        for param in model.parameters():
            if param.grad is not None:
                grad_norm += float(param.grad.detach().norm().cpu())
        print(
            "matrix_loss: {:.6f}, matrix_grad_norm: {:.6e}".format(
                loss.item(),
                grad_norm,
            )
        )

    optimizer.step()
    return gen_mat
