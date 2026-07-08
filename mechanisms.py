import copy
import math

import torch

import quantizer
import utils
from DNNQuantize import PyTorchMLP, train_model


def get_fixed_lattice_matrix(name, device=None):
    """Return a fixed lattice generator matrix."""
    name = str(name).lower()

    if name in ["fixed-hexagon", "hexagon", "hex"]:
        matrix = torch.tensor(
            [[1.0, 0.5], [0.0, math.sqrt(3.0) / 2.0]],
            dtype=torch.float32,
            device=device,
        )

    elif name in ["fixed-a2", "a2"]:
        matrix = torch.tensor(
            [[math.sqrt(2.0), 0.0], [-1.0 / math.sqrt(2.0), math.sqrt(3.0 / 2.0)]],
            dtype=torch.float32,
            device=device,
        )

    elif name in ["fixed-d2", "d2"]:
        matrix = torch.tensor(
            [[2.0, 0.0], [1.0, -1.0]],
            dtype=torch.float32,
            device=device,
        )

    elif name in ["fixed-e8", "e8"]:
        e8_rows = torch.tensor(
            [
                [2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [-1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, -1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, -1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, -1.0, 1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, -1.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 1.0, 0.0],
                [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
            ],
            dtype=torch.float32,
            device=device,
        )
        # The quantizer generates points as integer_coefficients @ matrix.T.
        matrix = e8_rows.T.contiguous()

    else:
        raise ValueError("Unknown fixed lattice: {}".format(name))

    return matrix


class IdentityMechanism:
    def __init__(self, args=None):
        self.args = args
        self.name = "none"

    def __call__(self, x, user_idx=None):
        del user_idx
        return x


class FixedLatticeMechanism:
    def __init__(self, args, name):
        self.args = args
        self.name = name
        utils.set_num_codewords_from_R(args)

        matrix = get_fixed_lattice_matrix(name, device=args.device)
        self.mechanism = quantizer.LatticeQuantization(args, matrix, True)

        print(
            "Using {}, R={}, lattice_dim={}, num_codewords={}, num_overloading={}".format(
                name,
                args.R,
                args.lattice_dim,
                args.num_codewords,
                args.num_overloading,
            )
        )

    def __call__(self, x, user_idx=None):
        del user_idx
        y, _ = self.mechanism(x, False, True)
        return y


class OLALaMechanism:
    def __init__(self, args):
        if getattr(args, "train_with_alpha", False):
            raise NotImplementedError("train_with_alpha is not included in this release.")

        self.args = args
        self.num_users = args.num_users
        self.device = args.device
        utils.set_num_codewords_from_R(args)

        self.matrix_models = []
        self.matrix_optimizers = []
        self.matrix_criteria = []
        self.user_matrices = []

        default_matrix = utils.default_matrix(args)

        for _ in range(self.num_users):
            matrix_model = PyTorchMLP(output_dim=args.lattice_dim).to(self.device)
            self.matrix_models.append(matrix_model)
            self.matrix_optimizers.append(torch.optim.Adam(matrix_model.parameters(), lr=args.matrix_lr))
            self.matrix_criteria.append(torch.nn.MSELoss())
            self.user_matrices.append(default_matrix.clone())

        print(
            "Using OLALa, R={}, lattice_dim={}, num_codewords={}, modulo_of_matrix={}, loss_by={}".format(
                args.R,
                args.lattice_dim,
                args.num_codewords,
                args.modulo_of_matrix,
                args.loss_by,
            )
        )

    def train_user_matrix(self, user_idx, local_model, global_model, train_loader, another_model=None):
        train_vec = utils.get_train_vector(local_model, global_model, self.args).to(self.device)

        if self.args.loss_by == "accuracy" and another_model is None:
            another_model = copy.deepcopy(local_model)

        matrix = train_model(
            model=self.matrix_models[user_idx],
            args=self.args,
            criterion=self.matrix_criteria[user_idx],
            testData=train_vec,
            optimizer=self.matrix_optimizers[user_idx],
            train_loader=train_loader,
            anotherModel=another_model,
        )

        if matrix is not None:
            self.user_matrices[user_idx] = matrix.detach().to(self.device)

        return self.user_matrices[user_idx]

    def __call__(self, x, user_idx=None):
        if user_idx is None:
            raise ValueError("OLALaMechanism requires user_idx.")

        matrix = self.user_matrices[user_idx].to(x.device)
        mechanism = quantizer.LatticeQuantization(self.args, matrix, True)
        y, _ = mechanism(x, False, True)
        return y


class StaticEachMechanism:
    def __init__(self, args):
        self.args = args
        self.device = args.device
        self.num_users = args.num_users
        self.fitted = False
        utils.set_num_codewords_from_R(args)

        self.matrix_models = []
        self.optimizers = []
        self.criteria = []
        self.user_matrices = []

        for _ in range(self.num_users):
            model = PyTorchMLP(output_dim=args.lattice_dim).to(self.device)
            self.matrix_models.append(model)
            self.optimizers.append(torch.optim.Adam(model.parameters(), lr=args.matrix_lr))
            self.criteria.append(torch.nn.MSELoss())
            self.user_matrices.append(utils.default_matrix(args).clone())

        print(
            "Using Static-Each, R={}, num_codewords={}, static_matrix_steps={}".format(
                args.R,
                args.num_codewords,
                args.static_matrix_steps,
            )
        )

    def fit_static_if_needed(self, local_models, global_model):
        if self.fitted:
            return

        for _ in range(int(self.args.static_matrix_steps)):
            for user_idx in range(self.num_users):
                user = local_models[user_idx]
                train_vec = utils.get_train_vector(user["model"], global_model, self.args).to(self.device)
                another_model = copy.deepcopy(user["model"]) if self.args.loss_by == "accuracy" else None

                matrix = train_model(
                    model=self.matrix_models[user_idx],
                    args=self.args,
                    criterion=self.criteria[user_idx],
                    testData=train_vec,
                    optimizer=self.optimizers[user_idx],
                    train_loader=user["data"],
                    anotherModel=another_model,
                )

                if matrix is not None:
                    self.user_matrices[user_idx] = matrix.detach().to(self.device)

        self.fitted = True

    def __call__(self, x, user_idx=None):
        if user_idx is None:
            raise ValueError("StaticEachMechanism requires user_idx.")

        matrix = self.user_matrices[user_idx].to(x.device)
        mechanism = quantizer.LatticeQuantization(self.args, matrix, True)
        y, _ = mechanism(x, False, True)
        return y


class StaticGlobalMechanism:
    def __init__(self, args):
        self.args = args
        self.device = args.device
        self.num_users = args.num_users
        self.fitted = False
        utils.set_num_codewords_from_R(args)

        self.matrix_model = PyTorchMLP(output_dim=args.lattice_dim).to(self.device)
        self.optimizer = torch.optim.Adam(self.matrix_model.parameters(), lr=args.matrix_lr)
        self.criterion = torch.nn.MSELoss()
        self.global_matrix = utils.default_matrix(args).clone()

        print(
            "Using Static-Global, R={}, num_codewords={}, static_matrix_steps={}".format(
                args.R,
                args.num_codewords,
                args.static_matrix_steps,
            )
        )

    def fit_static_if_needed(self, local_models, global_model):
        if self.fitted:
            return

        for _ in range(int(self.args.static_matrix_steps)):
            for user_idx in range(self.num_users):
                user = local_models[user_idx]
                train_vec = utils.get_train_vector(user["model"], global_model, self.args).to(self.device)
                another_model = copy.deepcopy(user["model"]) if self.args.loss_by == "accuracy" else None

                matrix = train_model(
                    model=self.matrix_model,
                    args=self.args,
                    criterion=self.criterion,
                    testData=train_vec,
                    optimizer=self.optimizer,
                    train_loader=user["data"],
                    anotherModel=another_model,
                )

                if matrix is not None:
                    self.global_matrix = matrix.detach().to(self.device)

        self.fitted = True

    def __call__(self, x, user_idx=None):
        del user_idx
        matrix = self.global_matrix.to(x.device)
        mechanism = quantizer.LatticeQuantization(self.args, matrix, True)
        y, _ = mechanism(x, False, True)
        return y


class QSGDIdentityMechanism:
    """Identity-lattice quantization baseline at the same scalar rate."""

    def __init__(self, args):
        self.args = args
        self.name = "qsgd"
        utils.set_num_codewords_from_R(args)

        matrix = torch.eye(args.lattice_dim, dtype=torch.float32, device=args.device)
        self.mechanism = quantizer.LatticeQuantization(args, matrix, True)

        print(
            "Using QSGD/identity lattice, R={}, lattice_dim={}, num_codewords={}, num_overloading={}".format(
                args.R,
                args.lattice_dim,
                args.num_codewords,
                args.num_overloading,
            )
        )

    def __call__(self, x, user_idx=None):
        del user_idx
        y, _ = self.mechanism(x, False, True)
        return y


class TopKMechanism:
    """Top-K sparsification matched to the target bit budget."""

    def __init__(self, args):
        self.args = args
        self.name = "topk"
        self.value_bits = int(getattr(args, "topk_value_bits", 32))
        self.use_error_feedback = bool(getattr(args, "topk_error_feedback", False))
        self.residuals = {}
        self._printed = False

        print(
            "Using Top-K, R={}, value_bits={}, error_feedback={}".format(
                args.R,
                self.value_bits,
                self.use_error_feedback,
            )
        )

    def _compute_k(self, n):
        index_bits = max(1, math.ceil(math.log2(max(2, n))))
        bits_per_kept = self.value_bits + index_bits
        k = int(math.floor(float(self.args.R) * n / bits_per_kept))
        k = max(1, min(k, n))
        actual_bpp = k * bits_per_kept / float(n)
        return k, index_bits, actual_bpp

    def __call__(self, x, user_idx=None):
        original_shape = x.shape
        flat = x.reshape(-1)
        n = flat.numel()

        if n == 0:
            return x

        if self.use_error_feedback and user_idx is not None:
            residual = self.residuals.get(user_idx)
            if residual is None or residual.numel() != n or residual.device != flat.device:
                residual = torch.zeros_like(flat)
            work = flat + residual
        else:
            work = flat

        k, index_bits, actual_bpp = self._compute_k(n)

        if not self._printed:
            print(
                "Top-K bit matching: n={}, R={}, k={}, index_bits={}, value_bits={}, actual_bpp={:.4f}".format(
                    n,
                    self.args.R,
                    k,
                    index_bits,
                    self.value_bits,
                    actual_bpp,
                )
            )
            self._printed = True

        _, idx = torch.topk(torch.abs(work), k=k, largest=True, sorted=False)
        sparse = torch.zeros_like(work)
        sparse[idx] = work[idx]

        if self.use_error_feedback and user_idx is not None:
            self.residuals[user_idx] = (work - sparse).detach()

        return sparse.reshape(original_shape)


def make_mechanism(args):
    name = str(args.mechanism).lower()

    if name in ["none", "no", "false", "0"]:
        return IdentityMechanism(args)

    if name in ["fixed-hexagon", "fixed_hexagon", "hexagon", "hex"]:
        return FixedLatticeMechanism(args, "Fixed-Hexagon")

    if name in ["fixed-a2", "fixed_a2", "a2"]:
        return FixedLatticeMechanism(args, "Fixed-A2")

    if name in ["fixed-d2", "fixed_d2", "d2"]:
        return FixedLatticeMechanism(args, "Fixed-D2")

    if name in ["olala", "adaptive"]:
        return OLALaMechanism(args)

    if name in ["static-each", "static_each", "staticeach"]:
        return StaticEachMechanism(args)

    if name in ["static-global", "static_global", "staticglobal"]:
        return StaticGlobalMechanism(args)

    if name in ["qsgd", "identity", "identity-lattice", "identity_lattice"]:
        return QSGDIdentityMechanism(args)

    if name in ["topk", "top-k", "top_k"]:
        return TopKMechanism(args)

    if name in ["fixed-e8", "fixed_e8", "e8"]:
        args.lattice_dim = 8
        utils.set_num_codewords_from_R(args)
        return FixedLatticeMechanism(args, "Fixed-E8")

    raise ValueError("Unknown mechanism: {}".format(args.mechanism))
