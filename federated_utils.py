import copy
import math

import torch
import torch.optim as optim

import utils


def federated_setup(global_model, train_data, args):
    """Create local models, optimizers, and data loaders for all clients."""
    indexes = torch.randperm(len(train_data))
    user_data_len = math.floor(len(train_data) / args.num_users)
    local_models = {}

    loader_kwargs = {
        "batch_size": args.train_batch_size,
        "shuffle": True,
        "num_workers": args.num_workers,
        "drop_last": args.drop_last,
    }
    if args.device == "cuda":
        loader_kwargs["pin_memory"] = args.pin_memory

    for user_idx in range(args.num_users):
        user_indices = utils.get_user_indices(
            train_data=train_data,
            user_idx=user_idx,
            args=args,
            indexes=indexes,
            user_data_len=user_data_len,
        )

        if getattr(args, "data_verbose", False):
            print("user {} samples: {}".format(user_idx, len(user_indices)))

        user_model = copy.deepcopy(global_model)
        user = {
            "data": torch.utils.data.DataLoader(
                torch.utils.data.Subset(train_data, user_indices),
                **loader_kwargs,
            ),
            "model": user_model,
        }

        if args.optimizer == "sgd":
            user["opt"] = optim.SGD(user_model.parameters(), lr=args.lr, momentum=args.momentum)
        else:
            user["opt"] = optim.Adam(user_model.parameters(), lr=args.lr)

        if args.lr_scheduler:
            user["scheduler"] = optim.lr_scheduler.ReduceLROnPlateau(
                user["opt"],
                patience=10,
                factor=0.1,
            )

        local_models[user_idx] = user

    return local_models, user_data_len


def distribute_model(local_models, global_model):
    global_state = copy.deepcopy(global_model.state_dict())
    for user_idx in range(len(local_models)):
        local_models[user_idx]["model"].load_state_dict(global_state)


def snr_ratio(original, reconstructed, eps=1e-12):
    return torch.var(original, unbiased=False) / (torch.var(original - reconstructed, unbiased=False) + eps)


def aggregate_models(local_models, global_model, mechanism, args=None, selected_users=None, user_data_len=None):
    """Aggregate compressed client models or client updates into the global model."""
    state_dict = copy.deepcopy(global_model.state_dict())
    keys = [name for name, param in global_model.named_parameters() if param.requires_grad]

    global_vec, shapes = utils.flatten_model(global_model, keys)
    sum_vec = None
    snr_users = []

    should_use_diff = False if args is None else getattr(args, "should_use_diff_in_quantizer", False)

    if selected_users is None:
        selected_users = list(range(len(local_models)))

    if user_data_len is None or isinstance(user_data_len, int):
        weights = {user_idx: 1.0 / len(selected_users) for user_idx in selected_users}
    else:
        total_data = sum(user_data_len[user_idx] for user_idx in selected_users)
        weights = {
            user_idx: float(user_data_len[user_idx]) / float(total_data)
            for user_idx in selected_users
        }

    for user_idx in selected_users:
        local_vec, _ = utils.flatten_model(local_models[user_idx]["model"], keys)
        vec_orig = local_vec - global_vec if should_use_diff else local_vec
        vec_compressed = mechanism(vec_orig, user_idx=user_idx)

        snr_users.append(snr_ratio(vec_orig, vec_compressed))
        weighted_vec = weights[user_idx] * vec_compressed
        sum_vec = weighted_vec if sum_vec is None else sum_vec + weighted_vec

    new_global_vec = global_vec + sum_vec if should_use_diff else sum_vec

    state_dict = utils.set_flat_vector_to_state_dict(
        state_dict=state_dict,
        flat_vector=new_global_vec,
        keys=keys,
        shapes=shapes,
    )
    global_model.load_state_dict(copy.deepcopy(state_dict))

    return sum(snr_users) / len(snr_users)
