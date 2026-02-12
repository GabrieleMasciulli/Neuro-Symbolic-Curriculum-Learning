# Module which contains the code for active learning training
import torch
import numpy as np

import wandb
import os

from utils.wandb_logger import *
from utils.status import progress_bar
from datasets.utils.base_dataset import BaseDataset, get_loader, BOIA_get_loader
from models.mnistdpl import MnistDPL
from utils.dpl_loss import ADDMNIST_DPL
from utils.metrics import (
    evaluate_metrics,
    evaluate_mix,
    mean_entropy,
    accuracy_binary,
)
from utils import fprint
from warmup_scheduler import GradualWarmupScheduler
from utils.instance_specific_risk import compute_instance_specific_risks_from_model


# ---------------------------------------------------------------------------
# Sample selection strategies
# ---------------------------------------------------------------------------
def select_random(unlabeled_indices: list, n_samples: int) -> list:
    """Randomly select samples from the unlabeled pool.

    Args:
        unlabeled_indices: list of indices not yet labeled.
        n_samples: number of samples to select.

    Returns:
        List of selected indices.
    """
    n_samples = min(n_samples, len(unlabeled_indices))
    selected = list(np.random.choice(unlabeled_indices, size=n_samples, replace=False))
    return selected


def select_instance_risk(unlabeled_indices: list, n_samples: int, model, loader, args) -> list:
    """Select samples based on risk / uncertainty (TODO: implement scoring).

    Args:
        unlabeled_indices: list of indices not yet labeled.
        n_samples: number of samples to select.
        model: the current model (can be used for uncertainty estimation).
        loader: DataLoader for the dataset.
        args: parsed arguments.

    Returns:
        List of selected indices.
    """ 
    print(f"\n--- Computing instance-specific risks ---")
    
    risks = compute_instance_specific_risks_from_model(
        model, loader, max_digit=9 if args.dataset == "shortmnist" else 4
    )

    # Sort all indices by risk in descending order (highest risk first)
    sorted_indices = np.argsort(risks)[::-1]

    # Filter to only unlabeled indices and select up to n_samples
    unlabeled_set = set(unlabeled_indices)
    selected = []
    for idx in sorted_indices:
        if int(idx) in unlabeled_set:
            selected.append(int(idx))
        if len(selected) >= n_samples:
            break

    return selected


def _train_cycle(
    model, train_loader, val_loader, _loss, args, n_epochs, scheduler, w_scheduler
):
    """Run training for a given number of epochs (one AL cycle).

    Returns:
        best_f1: best validation F1 achieved during this cycle.
    """
    best_f1 = 0.0

    for epoch in range(n_epochs):
        model.train()
        ys, y_true, cs, cs_true = None, None, None, None

        for i, data in enumerate(train_loader):
            images, labels, concepts = data

            # Handle contrastive pairs
            if args.contrastive and isinstance(images, (list, tuple)):
                view1, view2 = images
                view1, view2 = view1.to(model.device), view2.to(model.device)
                out_dict = model(torch.cat([view1, view2]))
                labels = torch.cat([labels, labels], dim=0)
                concepts = torch.cat([concepts, concepts], dim=0)
            else:
                images = images.to(model.device)
                out_dict = model(images)

            labels, concepts = (
                labels.to(model.device),
                concepts.to(model.device),
            )

            out_dict.update({"INPUTS": images, "LABELS": labels, "CONCEPTS": concepts})

            model.opt.zero_grad()
            loss, losses = _loss(out_dict, args)
            loss.backward()
            model.opt.step()

            # Accumulate predictions
            if ys is None:
                ys = out_dict["YS"]
                y_true = out_dict["LABELS"]
                cs = out_dict["pCS"]
                cs_true = out_dict["CONCEPTS"]
            else:
                ys = torch.concatenate((ys, out_dict["YS"]), dim=0)
                y_true = torch.concatenate((y_true, out_dict["LABELS"]), dim=0)
                cs = torch.concatenate((cs, out_dict["pCS"]), dim=0)
                cs_true = torch.concatenate((cs_true, out_dict["CONCEPTS"]), dim=0)

            if not args.tuning and args.wandb is not None:
                wandb_log_step(i, epoch, loss.item(), losses)

            if i % 10 == 0:
                progress_bar(i, len(train_loader) - 9, epoch, loss.item())

        # --- Epoch-level train metrics ---
        if args.task == "boia":
            acc, f1_train = accuracy_binary(ys, y_true)
            print(f"\n  Train Label acc: {acc}, Train Label f1: {f1_train}")
        else:
            y_pred = torch.argmax(ys, dim=-1)
            acc = (
                (y_pred.detach().cpu() == y_true.detach().cpu()).sum().item()
                / len(y_true)
                * 100
            )
            print(f"\n  Train acc: {acc:.2f}%  ({len(y_true)} samples)")

        # --- Validation ---
        model.eval()
        tloss, cacc, yacc, f1 = evaluate_metrics(model, val_loader, args)

        # LR scheduling
        if epoch < args.warmup_steps:
            w_scheduler.step()
        else:
            scheduler.step()

        fprint(f"  ACC C {cacc:.2f}  ACC Y {yacc:.2f}  F1 Y {f1:.4f}")

        if f1 > best_f1:
            best_f1 = f1

        if not args.tuning and args.wandb is not None:
            wandb_log_epoch(
                epoch=epoch,
                acc=yacc,
                cacc=cacc,
                tloss=tloss,
                lr=float(scheduler.get_last_lr()[0]),
            )

    return best_f1


def train_active(model: MnistDPL, dataset: BaseDataset, _loss: ADDMNIST_DPL, args):
    """Active learning training loop.

    At each cycle:
      1. Select ``al_query_size`` new samples from the unlabeled pool.
      2. Give concept supervision to all selected samples so far.
      3. Recreate the train loader and train.

    Args:
        model: the model to train.
        dataset: the dataset object (must implement ``give_supervision_to``).
        _loss: the loss function.
        args: parsed command line arguments.
    """
    active_type = args.active_type
    al_cycles = args.al_cycles
    al_query_size = args.al_query_size
    epochs_per_cycle = args.n_epochs

    save_path = (
        f"./checkpoints/best_model_{args.dataset}_{args.model}_{args.seed}"
        f"_active-{active_type}.pth"
    )

    # --- Setup ---
    # Store initial model state for resetting each cycle
    initial_state_dict = model.state_dict().copy()
    
    model.to(model.device)
    if args.dataset == "shortmnist":
        model = model.float()

    train_loader, val_loader, test_loader = dataset.get_data_loaders()
    dataset.print_stats()

    n_train = len(dataset.dataset_train)
    unlabeled_indices = list(range(n_train))
    labeled_indices = []

    best_f1_global = 0.0

    fprint("\n--- Start of Active Learning ---\n")
    fprint(f"  Strategy: {active_type}")
    fprint(
        f"  Cycles: {al_cycles}, Query size: {al_query_size}, "
        f"Epochs/cycle: {epochs_per_cycle}"
    )
    fprint(f"  Total train samples: {n_train}\n")

    # --- Active Learning Loop ---
    for cycle in range(al_cycles + 1): # include initial cycle with 0 supervision
        fprint(f"\n{'='*60}")
        fprint(f"  Active Learning Cycle {cycle}/{al_cycles}")
        fprint(f"{'='*60}")

        # --- Sample selection (skip first cycle: train with 0 supervision) ---
        if cycle == 0:
            fprint(f"  Cycle 1: Training with 0 supervised samples")
        else:
            # For instance_risk, load best model from previous cycle
            if active_type == "instance_risk" and os.path.exists(save_path):
                model.load_state_dict(torch.load(save_path))
                model.eval()
                fprint(f"  Loaded best model from previous cycle for risk computation")

            # select new samples from unlabeled pool
            if active_type == "random":
                selected = select_random(unlabeled_indices, al_query_size)
            elif active_type == "instance_risk":
                selected = select_instance_risk(
                    unlabeled_indices, al_query_size, model, train_loader, args
                )
            # todo: implement class-specific risk selection strategy

            # move selected samples from unlabeled → labeled (no duplicates)
            labeled_indices.extend(selected)
            for idx in selected:
                unlabeled_indices.remove(idx)

            fprint(f"  Selected {len(selected)} new samples")

            # Debug: print concept distribution of selected samples
            all_real_concepts = dataset.dataset_train.real_concepts[labeled_indices]
            concept_counts = {}
            for concept_pair in all_real_concepts:
                for c in concept_pair:
                    c_int = int(c)
                    concept_counts[c_int] = concept_counts.get(c_int, 0) + 1
            print(f"\n=== Cycle {cycle} Concept Summary (all labeled samples) ===")
            for concept_id, count in sorted(concept_counts.items()):
                print(f"  Concept {concept_id}: {count} samples")
            print(f"  Total unique concepts: {len(concept_counts)} / "
                  f"{10 if args.dataset == 'shortmnist' else 5}")
            print("=" * 50)

        fprint(
            f"  Labeled: {len(labeled_indices)} | "
            f"Unlabeled: {len(unlabeled_indices)}"
        )

        # --- Wandb ---
        if not args.tuning and args.wandb is not None:
            run_name = f"{args.dataset}-{args.model}-seed{args.seed}-active-{active_type}-cycle{cycle}-samples{len(labeled_indices)}"
            fprint("\n---wandb on\n")
            wandb.init(
                project=args.project,
                group=args.group_name,
                name=run_name,
                config=args,
            )

        # Reset model weights to initial state for training
        model.load_state_dict(initial_state_dict)
        fprint(f"  Model weights reset to initial state")

        # update concept supervision on the dataset
        dataset.give_supervision_to(labeled_indices)

        # recreate train loader with updated supervision
        if args.dataset in ["boia"]:
            train_loader = BOIA_get_loader(
                dataset.dataset_train, args.batch_size, val_test=False
            )
        else:
            train_loader = get_loader(
                dataset.dataset_train, args.batch_size, val_test=False
            )

        # re-initialize optimizer and schedulers for this cycle
        model.start_optim(args)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(model.opt, args.exp_decay)
        w_scheduler = None
        if args.warmup_steps > 0:
            w_scheduler = GradualWarmupScheduler(model.opt, 1.0, args.warmup_steps)
        model.opt.zero_grad()
        model.opt.step()

        # train for this cycle
        cycle_best_f1 = _train_cycle(
            model,
            train_loader,
            val_loader,
            _loss,
            args,
            n_epochs=epochs_per_cycle,
            scheduler=scheduler,
            w_scheduler=w_scheduler,
        )

        # save best model across cycles
        if cycle_best_f1 > best_f1_global:
            best_f1_global = cycle_best_f1
            torch.save(model.state_dict(), save_path)
            fprint(f"  New best model saved (F1: {best_f1_global:.4f})")

        # log cycle-level metrics to wandb
        if not args.tuning and args.wandb is not None:
            wandb.log(
                {
                    "al_cycle": cycle,
                    "al_labeled_samples": len(labeled_indices),
                    "al_cycle_best_f1": cycle_best_f1,
                    "al_global_best_f1": best_f1_global,
                }
            )

            # --- Final Evaluation ---
            fprint(f"\n{'='*60}")

            # Load best model and evaluate on test
            if os.path.exists(save_path):
                model.load_state_dict(torch.load(save_path))

            model.eval()
            y_true, c_true, y_pred, c_pred, p_cs, p_ys, p_cs_all, p_ys_all = evaluate_metrics(
                model, test_loader, args, last=True
            )

            yac, yf1 = evaluate_mix(y_true, y_pred)
            cac, cf1 = evaluate_mix(c_true, c_pred)
            h_c = mean_entropy(p_cs_all, model.n_facts)

            fprint(f"Test Concepts:  ACC: {cac}, F1: {cf1}")
            fprint(f"Test Labels:    ACC: {yac}, F1: {yf1}")
            fprint(f"Test Entropy:   H(C): {h_c}")

            if not args.tuning and args.wandb is not None:
                wandb.log({"test-y-acc": yac * 100, "test-y-f1": yf1 * 100})
                wandb.log({"test-c-acc": cac * 100, "test-c-f1": cf1 * 100})
                wandb.finish()
    
    # --- End of Active Learning ---
    fprint(f"  Active Learning Complete")
    fprint(
        f"  Best F1: {best_f1_global:.4f} with {len(labeled_indices)} labeled samples"
    )
    fprint(f"{'='*60}\n")