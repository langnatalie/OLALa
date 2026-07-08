import io
import os
import random
import time
from collections import defaultdict
from statistics import mean

import numpy as np
import torch
from PIL import Image
from tensorboardX import SummaryWriter
from torch.utils.data import Dataset
from torchvision import datasets, transforms


def sync_cuda_if_needed(device):
    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize()


def sample_clients(args, epoch):
    """Select participating clients for a communication round."""
    if not hasattr(args, "clients_per_round") or args.clients_per_round == -1:
        return list(range(args.num_users))

    if args.clients_per_round > args.num_users:
        raise ValueError(
            "clients_per_round={} cannot be larger than num_users={}".format(
                args.clients_per_round,
                args.num_users,
            )
        )

    rng = np.random.RandomState(args.seed + epoch)
    selected_users = rng.choice(args.num_users, size=args.clients_per_round, replace=False)
    return selected_users.tolist()


def get_writer_ids_from_train_data(train_data):
    if hasattr(train_data, "writer_ids"):
        return list(train_data.writer_ids)

    if isinstance(train_data, torch.utils.data.Subset):
        parent_dataset = train_data.dataset
        subset_indices = train_data.indices

        if hasattr(parent_dataset, "writer_ids"):
            return [parent_dataset.writer_ids[parent_idx] for parent_idx in subset_indices]

    raise ValueError("FEMNIST data has no writer_ids.")


def get_femnist_user_indices(train_data, num_users):
    writer_ids = get_writer_ids_from_train_data(train_data)
    writer_to_indices = defaultdict(list)

    for local_idx, writer_id in enumerate(writer_ids):
        writer_to_indices[writer_id].append(local_idx)

    writers = sorted(
        writer_to_indices.keys(),
        key=lambda w: len(writer_to_indices[w]),
        reverse=True,
    )

    if num_users > len(writers):
        raise ValueError(
            "Requested {} users, but only {} FEMNIST writers are available.".format(
                num_users,
                len(writers),
            )
        )

    selected_writers = writers[:num_users]
    user_indices = []
    user_data_len = []

    for writer_id in selected_writers:
        indices = [int(i) for i in writer_to_indices[writer_id]]
        user_indices.append(indices)
        user_data_len.append(len(indices))

    return user_indices, user_data_len


class FEMNISTDataset(Dataset):
    classes = list(range(62))

    def __init__(self, hf_dataset, transform=None):
        self.hf_dataset = hf_dataset
        self.transform = transform
        self.classes = list(range(62))
        self.targets = [int(x) for x in self.hf_dataset["character"]]
        self.labels = self.targets
        self.writer_ids = list(self.hf_dataset["writer_id"])

    def __len__(self):
        return len(self.hf_dataset)

    @staticmethod
    def _decode_image(image):
        if isinstance(image, Image.Image):
            return image.convert("L")

        if isinstance(image, dict):
            if image.get("bytes") is not None:
                return Image.open(io.BytesIO(image["bytes"])).convert("L")
            if image.get("path") is not None:
                return Image.open(image["path"]).convert("L")
            if image.get("array") is not None:
                return Image.fromarray(np.array(image["array"])).convert("L")
            raise TypeError("Unknown FEMNIST image dict format: {}".format(image.keys()))

        if isinstance(image, np.ndarray):
            return Image.fromarray(image).convert("L")

        raise TypeError("Unsupported FEMNIST image type: {}".format(type(image)))

    def __getitem__(self, idx):
        item = self.hf_dataset[int(idx)]
        image = self._decode_image(item["image"])
        label = int(item["character"])

        if self.transform is not None:
            image = self.transform(image)

        return image, label


def data(args):
    if args.data == "mnist":
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ])
        train_data = datasets.MNIST(args.data_root, train=True, download=True, transform=transform)
        test_dataset = datasets.MNIST(args.data_root, train=False, download=True, transform=transform)

    elif args.data == "femnist":
        from collections import Counter
        from datasets import load_dataset

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.9637,), (0.1592,)),
        ])

        full_data = load_dataset("flwrlabs/femnist", split="train")
        all_writer_ids = full_data["writer_id"]
        writer_counts = Counter(all_writer_ids)
        selected_writers = [writer_id for writer_id, _ in writer_counts.most_common(args.num_users)]

        if getattr(args, "data_verbose", False):
            print("Loaded FEMNIST with {} samples".format(len(full_data)))
            print("Selected writers:", selected_writers)
            print("Selected writer sizes:", [writer_counts[w] for w in selected_writers])

        selected_set = set(selected_writers)
        writer_to_indices = defaultdict(list)
        for idx, writer_id in enumerate(all_writer_ids):
            if writer_id in selected_set:
                writer_to_indices[writer_id].append(idx)

        rng = np.random.RandomState(args.seed)
        train_indices = []
        test_indices = []

        for writer_id in selected_writers:
            indices = writer_to_indices[writer_id]
            rng.shuffle(indices)
            n_test = max(1, int(0.2 * len(indices)))
            test_indices.extend(indices[:n_test])
            train_indices.extend(indices[n_test:])

        train_data = FEMNISTDataset(full_data.select(train_indices), transform=transform)
        test_dataset = FEMNISTDataset(full_data.select(test_indices), transform=transform)

    else:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
        train_data = datasets.CIFAR10(args.data_root, train=True, download=True, transform=transform)
        test_dataset = datasets.CIFAR10(args.data_root, train=False, download=True, transform=transform)

    loader_kwargs = {
        "batch_size": args.test_batch_size,
        "shuffle": False,
        "num_workers": args.num_workers,
    }
    if args.device == "cuda":
        loader_kwargs["pin_memory"] = args.pin_memory

    test_loader = torch.utils.data.DataLoader(test_dataset, **loader_kwargs)
    return train_data, test_loader


def data_split(data, amount, args):
    generator = torch.Generator()
    generator.manual_seed(args.seed)

    train_len = len(data) - amount
    train_data, val_data = torch.utils.data.random_split(
        data,
        [train_len, amount],
        generator=generator,
    )

    loader_kwargs = {
        "batch_size": args.test_batch_size,
        "shuffle": False,
        "num_workers": args.num_workers,
    }
    if args.device == "cuda":
        loader_kwargs["pin_memory"] = args.pin_memory

    val_loader = torch.utils.data.DataLoader(val_data, **loader_kwargs)

    in_channels, dim1, dim2 = data[0][0].shape
    input_size = dim1 * dim2 if args.model in ["mlp", "linear"] else in_channels
    output_size = 62 if args.data == "femnist" else len(data.classes)

    return input_size, output_size, train_data, val_loader


def get_targets_from_train_data(train_data):
    if hasattr(train_data, "targets"):
        return torch.as_tensor(train_data.targets)

    if hasattr(train_data, "labels"):
        return torch.as_tensor(train_data.labels)

    if isinstance(train_data, torch.utils.data.Subset):
        parent_dataset = train_data.dataset
        subset_indices = train_data.indices

        if hasattr(parent_dataset, "targets"):
            return torch.as_tensor(parent_dataset.targets)[subset_indices]

        if hasattr(parent_dataset, "labels"):
            return torch.as_tensor(parent_dataset.labels)[subset_indices]

        targets = []
        for idx in subset_indices:
            _, label = parent_dataset[idx]
            targets.append(int(label))
        return torch.tensor(targets)

    try:
        targets = []
        for i in range(len(train_data)):
            _, label = train_data[i]
            targets.append(int(label))
        return torch.tensor(targets)
    except Exception as exc:
        raise ValueError("Parent dataset has no targets/labels.") from exc


def get_user_indices(train_data, user_idx=None, args=None, user_data_len=None, indexes=None, **kwargs):
    del kwargs

    if args is not None and args.data == "femnist":
        all_user_indices, all_user_data_len = get_femnist_user_indices(train_data, args.num_users)
        if user_idx is None:
            return all_user_indices, all_user_data_len
        return all_user_indices[user_idx]

    if args.noniid == -1:
        start = user_idx * user_data_len
        end = (user_idx + 1) * user_data_len
        return indexes[start:end]

    targets = get_targets_from_train_data(train_data).long()
    num_classes = 10
    classes_per_user = args.noniid

    if num_classes % args.num_users != 0:
        raise ValueError("This split assumes num_classes is divisible by num_users.")

    step = num_classes // args.num_users
    user_classes = [(user_idx * step + j) % num_classes for j in range(classes_per_user)]

    mask = torch.zeros(len(targets), dtype=torch.bool)
    for cls in user_classes:
        mask |= targets == cls

    user_indices = torch.nonzero(mask).view(-1)

    if getattr(args, "data_verbose", False):
        print("user {} classes {} num_samples {}".format(user_idx, user_classes, len(user_indices)))

    return user_indices


def train_one_epoch(
    train_loader,
    model,
    optimizer,
    creterion,
    device,
    iterations,
    args=None,
    user_idx=None,
    global_model=None,
    mechanism=None,
    another_model=None,
):
    model.train()
    losses = []
    local_iteration = 0

    for data_batch, label in train_loader:
        data_batch = data_batch.to(device, non_blocking=True)
        label = label.to(device, non_blocking=True)

        output = model(data_batch)
        loss = creterion(output, label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
        local_iteration += 1

        should_update_lattice = (
            mechanism is not None
            and hasattr(mechanism, "train_user_matrix")
            and args is not None
            and user_idx is not None
            and global_model is not None
            and local_iteration % args.modulo_of_matrix == 0
        )

        if should_update_lattice:
            do_timing = (
                getattr(args, "measure_timings", False)
                and getattr(args, "measure_online_matrix_this_epoch", False)
            )

            if do_timing:
                sync_cuda_if_needed(device)
                online_start = time.perf_counter()

            mechanism.train_user_matrix(
                user_idx=user_idx,
                local_model=model,
                global_model=global_model,
                train_loader=train_loader,
                another_model=another_model,
            )

            if do_timing:
                sync_cuda_if_needed(device)
                online_end = time.perf_counter()
                elapsed = online_end - online_start
                args.online_matrix_training_time += elapsed
                args.online_matrix_training_calls += 1
                args.online_matrix_training_time_by_user[user_idx] += elapsed

        if iterations is not None and local_iteration >= iterations:
            break

    return mean(losses) if losses else 0.0


def flatten_model(model, keys):
    state = model.state_dict()
    pieces = []
    shapes = []

    for key in keys:
        tensor = state[key]
        pieces.append(tensor.reshape(-1))
        shapes.append(tensor.shape)

    return torch.cat(pieces), shapes


def set_flat_vector_to_state_dict(state_dict, flat_vector, keys, shapes):
    pointer = 0

    for key, shape in zip(keys, shapes):
        numel = state_dict[key].numel()
        piece = flat_vector[pointer:pointer + numel]
        state_dict[key] = piece.reshape(shape).to(state_dict[key].dtype)
        pointer += numel

    return state_dict


def set_num_codewords_from_R(args):
    args.num_codewords = int(round(2 ** (args.lattice_dim * args.R)))


def default_matrix(args):
    if int(args.lattice_dim) == 2:
        return torch.tensor(
            [[0.9793, -0.9523], [-0.9842, -0.9561]],
            dtype=torch.float32,
            device=args.device,
        )

    return torch.eye(int(args.lattice_dim), dtype=torch.float32, device=args.device)


def get_train_vector(local_model, global_model, args):
    keys = [name for name, param in global_model.named_parameters() if param.requires_grad]
    local_vec, _ = flatten_model(local_model, keys)
    global_vec, _ = flatten_model(global_model, keys)

    if getattr(args, "should_use_diff_in_quantizer", False):
        return local_vec - global_vec

    return local_vec


def test(test_loader, model, creterion, device):
    model.eval()
    correct = 0

    with torch.no_grad():
        for data_batch, label in test_loader:
            data_batch = data_batch.to(device, non_blocking=True)
            label = label.to(device, non_blocking=True)
            output = model(data_batch)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(label.view_as(pred)).sum().item()

    return 100.0 * correct / len(test_loader.dataset)


def initializations(args):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)

    os.makedirs("checkpoints/{}".format(args.exp_name), exist_ok=True)

    boardio = SummaryWriter(log_dir="checkpoints/{}".format(args.exp_name))
    textio = IOStream("checkpoints/{}/run.log".format(args.exp_name))

    best_val_acc = -float("inf")
    path_best_model = "checkpoints/{}/model.best.t7".format(args.exp_name)

    return boardio, textio, best_val_acc, path_best_model


class IOStream:
    def __init__(self, path):
        self.f = open(path, "a", encoding="utf-8")

    def cprint(self, text):
        print(text)
        self.f.write(str(text) + "\n")
        self.f.flush()

    def close(self):
        self.f.close()
