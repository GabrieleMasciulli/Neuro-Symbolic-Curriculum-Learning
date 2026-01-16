"""
Class-Specific Risk Computation Module

This module contains utilities to compute class-specific risk bounds
from partial risk observations using constrained optimization.
"""

import numpy as np
import torch
from scipy.optimize import minimize
from sklearn.metrics import accuracy_score
from tqdm import tqdm


def get_single_concept_gold_priors(hidden_labels, num_classes):
    """
    Computes the exact marginal distribution (r) from the hidden gold labels.
    
    Args:
        hidden_labels: A list or numpy array of all ground truth concept in your dataset.
        num_classes: The total number of unique classes (c).
        
    Returns:
        priors: A numpy array of shape (c,) summing to 1.
    """
    # count occ. of each class
    counts = np.bincount(hidden_labels, minlength=num_classes)
    
    # normalize for probabilities
    priors = counts / np.sum(counts)
    
    return priors


def get_sigma_matrix(num_classes, priors, symbolic_function):
    """
    Constructs the sigma matrix (Σ_σ,r) described in Equation (1) which depends on the class priors.
    
    Args:
        num_classes: Number of concept classes
        priors: Prior distribution over classes
        symbolic_function: Function that maps concept pairs to weak labels (e.g., addition)
        
    Returns:
        Sigma: The sigma matrix of shape (dim, dim) where dim = num_classes^2
    """
    dim = num_classes ** 2
    Sigma = np.zeros((dim, dim))
    
    # iterate over all possible gold label pairs (i, j)
    for i in range(num_classes):
        for j in range(num_classes):
            # probabilty of this gold label pair occurring (assuming independence) i.e. P(Y=i) * P(Y=j)
            prob_gold = priors[i] * priors[j]
            
            # the correct weak label for this gold label pair
            s_true = symbolic_function(i, j)
            
            for i_prime in range(num_classes):
                for j_prime in range(num_classes):
                    s_pred = symbolic_function(i_prime, j_prime)  # s' = σ(i', j')
                    
                    # if weak labels differ, this contributes to the Partial Risk
                    if s_true != s_pred:
                        # u corresponds to confusion H[i, i_prime]
                        # v corresponds to confusion H[j, j_prime]
                        u = i * num_classes + i_prime
                        v = j * num_classes + j_prime
                        
                        Sigma[u, v] += prob_gold
                        
    return Sigma


def compute_class_specific_risk_bound(
    target_class_index,
    observed_partial_risk,
    num_classes,
    Sigma
):
    """
    Solves optimization program (2) to find the worst-case risk bound R_j(f).
    
    Args:
        target_class_index: The class index j for which to compute the risk bound
        observed_partial_risk: The observed partial risk R_P from validation
        num_classes: Number of concept classes
        Sigma: The sigma matrix computed from get_sigma_matrix
        
    Returns:
        risk_bound: Upper bound on the risk for the target class, or None if optimization fails
    """
    dim = num_classes ** 2
    
    # Objective: Maximize risk R_j(f)
    # since scipy only does minimization, we minimize the 'Probability Correct' instead i.e.
    # Risk = 1 - Probability Correct
    def objective(h):
        # the diagonal entry H[j, j] corresponds to P(Pred=j | Gold=j)
        # in the flattended vector h, this is at index (j * num_classes + j)
        correct_pred_idx = target_class_index * num_classes + target_class_index
        prob_correct = h[correct_pred_idx]
        
        return prob_correct

    # Constraint 1: Partial Risk Match (Eq 2, first constraint)
    # h^T * Sigma * h = R_P
    def constraint_partial_risk(h):
        return (h.T @ Sigma) @ h - observed_partial_risk
    
    # Constraint 2: Normalization (Eq 2, third constraint)
    # Each row of H must sum to 1
    constraints = [{'type': 'eq', 'fun': constraint_partial_risk}]
    
    for r in range(num_classes):
        def row_sum_constraint(h, row_idx=r):
            start = row_idx * num_classes
            end = start + num_classes
            return np.sum(h[start:end]) - 1.0
        constraints.append({'type': 'eq', 'fun': row_sum_constraint})

    # Bounds: Probabilities must be between 0 and 1 (Eq 2, second constraint)
    bounds = [(0, 1) for _ in range(dim)]
    
    # Initial guess: Identity matrix
    h0 = np.eye(num_classes).flatten()
    
    # Optimization
    result = minimize(
        objective, 
        h0, 
        method='SLSQP', 
        bounds=bounds,
        constraints=constraints,
        options={'maxiter': 1000, 'ftol': 1e-6}
    )
    
    if not result.success:
        print(f"Warning: Optimization failed for class {target_class_index}")
        return None
        
    # Convert minimized Prob_Correct back to Risk
    return 1.0 - result.fun


def get_concepts_and_labels_mnist(
    out_labels, out_concepts, true_concepts, dataset_name, is_ood=False
):
    """
    Extract predicted labels and concepts from model outputs for MNIST-based datasets.
    
    Args:
        out_labels: Model output logits for labels
        out_concepts: Model output logits/probabilities for concepts
        true_concepts: Ground truth concepts
        dataset_name: Name of the dataset
        is_ood: Whether this is out-of-distribution evaluation
        
    Returns:
        predicted_labels, predicted_concepts, refactored_true_concepts
    """
    label_logits = out_labels
    
    if not is_ood and dataset_name.lower() in ["shortmnist", "clipshortmnist", "shortmnist"]:
        label_logits = label_logits.clone()
        allowed = torch.tensor([6, 10, 12], device=label_logits.device)
        disallowed = torch.ones(label_logits.size(1), dtype=torch.bool, device=label_logits.device)
        disallowed[allowed] = False
        label_logits[:, disallowed] = 0

    predicted_labels = torch.argmax(label_logits, dim=-1)
    predicted_concepts = torch.argmax(out_concepts, dim=-1)

    predicted_concepts = predicted_concepts.reshape(-1)  # from [batch_size, 2] to [batch_size * 2]
    refactored_true_concepts = true_concepts.reshape(-1)  # from [batch_size, 2] to [batch_size * 2]

    return predicted_labels, predicted_concepts, refactored_true_concepts


def compute_class_specific_risks_from_model(
    model, 
    data_loader, 
    dataset_name,
    num_classes=10,
    symbolic_function=None,
    verbose=False
):
    """
    Compute class-specific risk bounds given a trained model and validation data.
    
    Args:
        model: The trained model
        data_loader: DataLoader for validation data
        dataset_name: Name of the dataset
        num_classes: Number of concept classes
        symbolic_function: Function mapping concept pairs to weak labels (default: addition)
        verbose: Whether to print progress
        
    Returns:
        class_specific_risks: Array of risk bounds for each class
        observed_partial_risk: The observed partial risk on validation set
    """
    if symbolic_function is None:
        # Default: addition for MNIST-based tasks
        symbolic_function = lambda y1, y2: y1 + y2
    
    model.eval()
    
    true_labels = []
    predicted_labels = []
    true_concepts = []
    
    iterator = tqdm(data_loader, desc="Computing risks") if verbose else data_loader
    
    with torch.no_grad():
        for data in iterator:
            images, labels, concepts = data
            images, labels, concepts = (
                images.to(model.device),
                labels.to(model.device),
                concepts.to(model.device),
            )
            
            out_dict = model(images)
            
            out_label, out_concept, concepts_flattened = get_concepts_and_labels_mnist(
                out_dict["YS"], out_dict["pCS"], concepts, dataset_name, is_ood=False
            )
            
            true_labels.append(labels.cpu().numpy())
            predicted_labels.append(out_label.detach().cpu().numpy())
            true_concepts.append(concepts_flattened.cpu().numpy())
    
    true_labels = np.concatenate(true_labels, axis=0)
    predicted_labels = np.concatenate(predicted_labels, axis=0)
    true_concepts = np.concatenate(true_concepts, axis=0)
    
    # Compute observed partial risk
    label_accuracy = accuracy_score(true_labels, predicted_labels)
    observed_partial_risk = 1 - label_accuracy
    
    if verbose:
        print(f"Observed Partial Risk (R_P): {observed_partial_risk:.4f}")
    
    # Compute gold priors from ground truth
    gold_priors = get_single_concept_gold_priors(true_concepts, num_classes)
    
    # Compute Sigma matrix
    Sigma = get_sigma_matrix(num_classes, gold_priors, symbolic_function)
    
    # Compute risk bounds for each class
    class_specific_risks = []
    for target_class in range(num_classes):
        risk_bound = compute_class_specific_risk_bound(
            target_class, 
            observed_partial_risk, 
            num_classes, 
            Sigma
        )
        
        class_specific_risks.append(risk_bound)
        
        if verbose:
            print(f"Upper bound risk for Class {target_class}: {risk_bound:.4f}")
    
    return np.array(class_specific_risks), observed_partial_risk
