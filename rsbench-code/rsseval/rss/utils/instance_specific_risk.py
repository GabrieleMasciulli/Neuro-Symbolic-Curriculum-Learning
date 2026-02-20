import numpy as np
import torch
from tqdm import tqdm


def get_valid_pairs(target_sum, max_digit=9):
    """
    Given a target sum, returns all valid pairs of digits that sum to the target.

    Args:
        target_sum: Integer target sum
        max_digit: Maximum digit value (default 9 for [0..9], use 4 for [0..4])
    """
    valid_pairs = []
    for y1 in range(min(max_digit + 1, target_sum + 1)):
        y2 = target_sum - y1
        if 0 <= y2 <= max_digit:
            valid_pairs.append((y1, y2))
    return valid_pairs


def compute_instance_specific_risks_from_model(model, dataloader, max_digit=9):
    """
    Compute instance-specific risk given a trained model and validation data.
    Designed for MNIST-style datasets with 2 single-class concepts and an
    addition constraint.

    Args:
        model: The model to evaluate
        dataloader: DataLoader containing the dataset
        max_digit: Maximum digit value (default 9 for [0..9], use 4 for [0..4])
    """
    instance_risks = []

    with torch.no_grad():
        for data in tqdm(dataloader, desc="Computing instance-specific risks"):
            images, labels, concepts = data
            images, labels, concepts = (
                images.to(model.device),
                labels.to(model.device),
                concepts.to(model.device),
            )

            out_dict = model(images)
            probs = torch.softmax(
                out_dict["pCS"], dim=-1
            )  # shape: (Batch, 2, 10) or (Batch, 2, 5)

            # compute Probability of Satisfaction for each sample
            batch_risks = []
            for i in range(len(labels)):
                s = labels[i].item()

                p1 = probs[i, 0]  # Prob vector for image 1
                p2 = probs[i, 1]  # Prob vector for image 2

                # Sum probability of all VALID pairs (y1, y2) such that y1+y2 = s
                prob_valid = 0.0
                valid_pairs = get_valid_pairs(
                    s, max_digit
                )  # e.g. for s=1, max_digit=4 returns [(0,1), (1,0)]

                for y1, y2 in valid_pairs:
                    prob_valid += (
                        p1[y1] * p2[y2] / len(valid_pairs)
                    )  # normalize by number of valid pairs

                # Risk = Probability of Constraint Violation
                batch_risks.append(1.0 - prob_valid.item())

            instance_risks.extend(batch_risks)

        return np.array(instance_risks)


def compute_multilabel_instance_risks_from_model(model, dataloader, verbose=True):
    """
    Compute instance-specific risks for a multilabel dataset (e.g. BOIA).

    For each sample the risk is the **average per-concept prediction error**:
        risk_i = (1 / C) * sum_j | p_j - gt_j |
    where p_j is the predicted probability for concept j and gt_j in {0, 1} is
    the ground-truth.  Low risk => the model is confident *and* correct on all
    concepts => easy sample.

    Args:
        model: The trained model (must output ``CS`` with sigmoid probabilities
               of shape ``(batch, num_concepts)``).
        dataloader: DataLoader for the dataset to evaluate.
        verbose: Whether to show a progress bar.

    Returns:
        instance_risks: 1-D numpy array of shape ``(num_samples,)`` with risk
                        scores in [0, 1].
    """
    instance_risks = []
    model.eval()

    iterator = (
        tqdm(dataloader, desc="Computing multilabel instance risks")
        if verbose
        else dataloader
    )

    with torch.no_grad():
        for data in iterator:
            images, labels, concepts = data
            images = images.to(model.device)
            concepts = concepts.to(model.device)

            out_dict = model(images)

            # CS contains sigmoid concept probabilities, shape (batch, C)
            concept_probs = out_dict["CS"]

            # Absolute prediction error per concept
            errors = torch.abs(concept_probs - concepts.float())  # (batch, C)
            # Mean error across concepts for each sample
            sample_risks = errors.mean(dim=-1)  # (batch,)
            instance_risks.extend(sample_risks.cpu().tolist())

    return np.array(instance_risks)
