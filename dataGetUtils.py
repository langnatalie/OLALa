import torch


def test(test_loader, model, criterion, device, should_keep_grads=False, verbose=False, shouldKeppGrads=None):
    """Evaluate a model and return accuracy and loss.

    The ``shouldKeppGrads`` argument is retained as a backward-compatible alias
    for older scripts.
    """
    if shouldKeppGrads is not None:
        should_keep_grads = shouldKeppGrads

    if should_keep_grads:
        model.eval()
        correct = 0
        test_loss = torch.tensor(0.0, device=device, requires_grad=True)

        for data, label in test_loader:
            data = data.to(device, non_blocking=True)
            label = label.to(device, non_blocking=True)
            output = model(data)
            test_loss = test_loss + criterion(output, label)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(label.view_as(pred)).sum().item()

        accuracy = 100.0 * correct / len(test_loader.dataset)
        test_loss = test_loss / len(test_loader.dataset)
    else:
        model.eval()
        correct = 0
        test_loss = 0.0

        with torch.no_grad():
            for data, label in test_loader:
                data = data.to(device, non_blocking=True)
                label = label.to(device, non_blocking=True)
                output = model(data)
                test_loss += criterion(output, label).item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(label.view_as(pred)).sum().item()

        accuracy = 100.0 * correct / len(test_loader.dataset)
        test_loss = test_loss / len(test_loader)

    if verbose:
        loss_value = test_loss.item() if torch.is_tensor(test_loss) else test_loss
        print("Accuracy: {:.2f}%".format(accuracy))
        print("Test Loss: {:.4f}".format(loss_value))

    return accuracy, test_loss


def getWeights(model):
    """Flatten all trainable parameters into one vector and record shapes."""
    flattened_weights = []
    shapes = {}

    for name, param in model.named_parameters():
        if param.requires_grad:
            weight = param.data.clone()
            shapes[name] = weight.shape
            flattened_weights.append(weight.reshape(-1))

    if not flattened_weights:
        raise ValueError("No trainable parameters found.")

    return torch.cat(flattened_weights), shapes


def restoreWeights(shapes, combined_weights, model):
    """Restore a flat vector into the trainable parameters of ``model``."""
    offset = 0
    restored_weights = {}

    for name, shape in shapes.items():
        num_elements = 1
        for s in shape:
            num_elements *= int(s)

        restored_weights[name] = combined_weights[offset:offset + num_elements].view(shape)
        offset += num_elements

    for name, param in model.named_parameters():
        if param.requires_grad and name in restored_weights:
            param.data = restored_weights[name].to(param.data.device, dtype=param.data.dtype)

    return model
