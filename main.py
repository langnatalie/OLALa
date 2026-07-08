import copy
import gc
import sys
import time
from collections import defaultdict
from statistics import mean

import numpy as np
import torch
from tqdm import tqdm

import federated_utils
import models
import utils
from configurations import args_parser
from mechanisms import make_mechanism


def sync_cuda_if_needed(args):
    if args.device == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize()


def build_model(args, input_size, output_size):
    if args.model == "mlp":
        return models.FC2Layer(input_size, output_size)
    if args.model == "cnn2":
        return models.CNN2Layer(input_size, output_size, args.data)
    if args.model == "cnn3":
        return models.CNN3Layer()
    if args.model == "mini_mobilenet_femnist":
        return models.MiniMobileNetFEMNIST(input_size, output_size)
    return models.Linear(input_size, output_size)


def main():
    start_time = time.time()
    args = args_parser()
    boardio, textio, best_val_acc, path_best_model = utils.initializations(args)

    textio.cprint(str(args))
    textio.cprint("device: {}".format(args.device))

    train_data, test_loader = utils.data(args)
    input_size, output_size, train_data, val_loader = utils.data_split(
        train_data,
        len(test_loader.dataset),
        args,
    )

    global_model = build_model(args, input_size, output_size).to(args.device)

    if args.print_model_summary:
        try:
            from torchinfo import summary
            textio.cprint(str(summary(global_model)))
        except Exception as exc:
            textio.cprint("Could not print model summary: {}".format(exc))

    train_criterion = torch.nn.CrossEntropyLoss(reduction="mean")
    test_criterion = torch.nn.CrossEntropyLoss(reduction="sum")

    if args.eval:
        global_model.load_state_dict(torch.load(path_best_model, map_location=args.device))
        test_acc = utils.test(test_loader, global_model, test_criterion, args.device)
        textio.cprint("eval test_acc: {:.2f}%".format(test_acc))
        gc.collect()
        sys.exit()

    local_models, user_data_len = federated_utils.federated_setup(global_model, train_data, args)
    mechanism = make_mechanism(args)

    train_loss_list = []
    val_acc_list = []
    test_acc_list = []
    snr_ratio_list = []
    best_test_acc = None

    for global_epoch in tqdm(range(args.global_epochs)):
        if getattr(args, "measure_timings", False):
            args.measure_online_matrix_this_epoch = global_epoch == args.measure_epoch
            args.online_matrix_training_time = 0.0
            args.online_matrix_training_calls = 0
            args.online_matrix_training_time_by_user = defaultdict(float)

            if args.measure_online_matrix_this_epoch:
                sync_cuda_if_needed(args)
                measured_round_start = time.perf_counter()
        else:
            args.measure_online_matrix_this_epoch = False

        federated_utils.distribute_model(local_models, global_model)
        users_loss = []
        selected_users = utils.sample_clients(args, global_epoch)

        print("epoch {} selected users: {}".format(global_epoch, selected_users))

        for user_idx in selected_users:
            local_models[user_idx]["model"].load_state_dict(copy.deepcopy(global_model.state_dict()))
            user = local_models[user_idx]
            user_loss = []

            for _ in range(args.local_epochs):
                train_loss = utils.train_one_epoch(
                    train_loader=user["data"],
                    model=user["model"],
                    optimizer=user["opt"],
                    creterion=train_criterion,
                    device=args.device,
                    iterations=args.local_iterations,
                    args=args,
                    user_idx=user_idx,
                    global_model=global_model,
                    mechanism=mechanism,
                )

                if args.lr_scheduler:
                    user["scheduler"].step(train_loss)

                user_loss.append(train_loss)

            users_loss.append(mean(user_loss))

        train_loss = mean(users_loss)

        if hasattr(mechanism, "fit_static_if_needed"):
            mechanism.fit_static_if_needed(local_models, global_model)

        snr_ratio = federated_utils.aggregate_models(
            local_models,
            global_model,
            mechanism,
            args=args,
            selected_users=selected_users,
            user_data_len=user_data_len,
        )

        if torch.isnan(snr_ratio):
            textio.cprint("Stop training: SNR is NaN")
            break

        snr_db = 10.0 * torch.log10(snr_ratio)
        snr_ratio_list.append(float(snr_ratio.detach().cpu()))

        val_acc = utils.test(val_loader, global_model, test_criterion, args.device)
        test_acc = utils.test(test_loader, global_model, test_criterion, args.device)

        train_loss_list.append(train_loss)
        val_acc_list.append(val_acc)
        test_acc_list.append(test_acc)

        boardio.add_scalar("train_loss", train_loss, global_epoch)
        boardio.add_scalar("val_acc", val_acc, global_epoch)
        boardio.add_scalar("test_acc", test_acc, global_epoch)
        boardio.add_scalar("snr_db", float(snr_db.detach().cpu()), global_epoch)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_test_acc = test_acc
            torch.save(global_model.state_dict(), path_best_model)

        avg_snr_ratio = sum(snr_ratio_list) / len(snr_ratio_list)
        avg_snr_db = 10.0 * np.log10(avg_snr_ratio)

        textio.cprint(
            "epoch: {:03d} | train_loss: {:.4f} | val_acc: {:.2f}% | "
            "test_acc: {:.2f}% | SNR: {:.2f} dB | avg_SNR: {:.2f} dB".format(
                global_epoch,
                train_loss,
                val_acc,
                test_acc,
                float(snr_db.detach().cpu()),
                avg_snr_db,
            )
        )

        gc.collect()
        if args.device == "cuda":
            torch.cuda.empty_cache()

        if getattr(args, "measure_timings", False) and getattr(args, "measure_online_matrix_this_epoch", False):
            sync_cuda_if_needed(args)
            measured_round_end = time.perf_counter()

            round_time = measured_round_end - measured_round_start
            online_total = args.online_matrix_training_time
            online_calls = args.online_matrix_training_calls

            if len(args.online_matrix_training_time_by_user) > 0:
                online_avg_per_user = sum(args.online_matrix_training_time_by_user.values()) / len(
                    args.online_matrix_training_time_by_user
                )
            else:
                online_avg_per_user = 0.0

            online_avg_per_call = online_total / max(online_calls, 1)
            online_percent_of_round = 100.0 * online_total / max(round_time, 1e-12)

            timing_msg = (
                "{}\n"
                "OLALa online matrix timing\n"
                "epoch: {}\n"
                "round_time: {:.6f} sec\n"
                "online_matrix_total: {:.6f} sec\n"
                "online_matrix_calls: {}\n"
                "online_matrix_avg_per_user: {:.6f} sec\n"
                "online_matrix_avg_per_call: {:.6f} sec\n"
                "online_matrix / round_time: {:.4f}%\n"
                "{}"
            ).format(
                "=" * 80,
                global_epoch,
                round_time,
                online_total,
                online_calls,
                online_avg_per_user,
                online_avg_per_call,
                online_percent_of_round,
                "=" * 80,
            )

            textio.cprint(timing_msg)
            sys.exit(0)

    out_dir = "checkpoints/{}".format(args.exp_name)
    np.save("{}/train_loss_list.npy".format(out_dir), np.array(train_loss_list))
    np.save("{}/val_acc_list.npy".format(out_dir), np.array(val_acc_list))
    np.save("{}/test_acc_list.npy".format(out_dir), np.array(test_acc_list))
    np.save("{}/snr_ratio_list.npy".format(out_dir), np.array(snr_ratio_list))

    elapsed_min = (time.time() - start_time) / 60.0
    textio.cprint("best_val_acc: {:.2f}%".format(best_val_acc))
    if best_test_acc is not None:
        textio.cprint("best_test_acc: {:.2f}%".format(best_test_acc))
    if test_acc_list:
        textio.cprint("last_test_acc: {:.2f}%".format(test_acc_list[-1]))
    textio.cprint("total execution time: {:.2f} min".format(elapsed_min))
    textio.close()


if __name__ == "__main__":
    main()
