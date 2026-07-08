import argparse


def str2bool(value):
    """Parse common string representations of booleans."""
    if isinstance(value, bool):
        return value

    value = value.lower()
    if value in ("yes", "true", "t", "1", "y"):
        return True
    if value in ("no", "false", "f", "0", "n"):
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")


def args_parser():
    parser = argparse.ArgumentParser(description="OLALa federated-learning experiments")

    # Experiment
    parser.add_argument("--exp_name", type=str, default="exp")
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--data_root", type=str, default="./data")
    parser.add_argument("--print_model_summary", type=str2bool, default=False)

    # Data
    parser.add_argument(
        "--data",
        type=str,
        default="mnist",
        choices=["mnist", "cifar10", "femnist"],
        help="Dataset to use.",
    )
    parser.add_argument(
        "--noniid",
        type=int,
        default=3,
        help="-1 means IID/even split. Positive k means each user receives k consecutive classes.",
    )
    parser.add_argument("--train_batch_size", type=int, default=16)
    parser.add_argument("--test_batch_size", type=int, default=1000)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--pin_memory", type=str2bool, default=True)
    parser.add_argument("--drop_last", type=str2bool, default=False)
    parser.add_argument("--data_verbose", type=str2bool, default=False)

    # Federated learning
    parser.add_argument(
        "--model",
        type=str,
        default="cnn2",
        choices=["cnn2", "cnn3", "mlp", "linear", "mini_mobilenet_femnist"],
    )
    parser.add_argument("--num_users", type=int, default=5)
    parser.add_argument(
        "--clients_per_round",
        type=int,
        default=-1,
        help="Number of randomly participating clients per round. Use -1 for all clients.",
    )
    parser.add_argument("--local_epochs", type=int, default=1)
    parser.add_argument("--local_iterations", type=int, default=100)
    parser.add_argument("--global_epochs", type=int, default=40)

    # Optimizer
    parser.add_argument("--optimizer", type=str, default="sgd", choices=["sgd", "adam"])
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--momentum", type=float, default=0.5)
    parser.add_argument("--lr_scheduler", type=str2bool, default=False)

    # Quantization / compression mechanisms
    parser.add_argument(
        "--mechanism",
        type=str,
        default="olala",
        choices=[
            "none",
            "hex", "a2", "d2", "e8",
            "Fixed-Hexagon", "Fixed-A2", "Fixed-D2",
            "static-each", "static-global",
            "olala",
            "qsgd", "topk",
        ],
        help="Compression mechanism used before FedAvg aggregation.",
    )
    parser.add_argument("--lattice_dim", type=int, default=2)
    parser.add_argument("--R", type=float, default=3, help="Rate in bits per scalar sample.")
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--num_overloading", type=float, default=-1)
    parser.add_argument("--lattice_grid_radius", type=int, default=2)
    parser.add_argument("--should_use_diff_in_quantizer", type=str2bool, default=False)
    parser.add_argument("--quantizer_verbose", type=str2bool, default=False)

    # Top-K baseline
    parser.add_argument(
        "--topk_value_bits",
        type=int,
        default=32,
        help="Number of bits used for each transmitted Top-K value. Index bits are counted separately.",
    )

    # Learned lattice
    parser.add_argument("--static_matrix_steps", type=int, default=1)
    parser.add_argument("--modulo_of_matrix", type=int, default=10)
    parser.add_argument("--matrix_lr", type=float, default=1e-6)
    parser.add_argument("--loss_by", type=str, default="mse", choices=["mse", "accuracy", "snr"])
    parser.add_argument(
        "--task_loss_batches",
        type=int,
        default=1,
        help="Mini-batches used to estimate the task-aware lattice loss. Use -1 for the full loader.",
    )
    parser.add_argument("--dnn_quantize_verbose", type=str2bool, default=False)

    # Deprecated option kept for compatibility with old command lines.
    parser.add_argument("--train_with_alpha", type=str2bool, default=False)

    # Timing measurement
    parser.add_argument(
        "--measure_timings",
        type=str2bool,
        default=False,
        help="If True, measure OLALa online matrix-training time for one global round and exit.",
    )
    parser.add_argument(
        "--measure_epoch",
        type=int,
        default=1,
        help="Global epoch used for timing measurement.",
    )

    args = parser.parse_args()

    if args.device == "cuda" and not __import__("torch").cuda.is_available():
        args.device = "cpu"

    # Paper convention: R = (1 / d) log_2(|codebook|).
    args.num_codewords = int(round(2 ** (args.lattice_dim * args.R)))

    return args
