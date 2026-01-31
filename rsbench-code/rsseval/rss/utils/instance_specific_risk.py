import numpy as np
import torch

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

def compute_instance_constraint_risk(model, dataloader, max_digit=9):
    """
    Computes the risk (probability of violating the constraint) 
    for each individual instance in the dataset.
    
    Args:
        model: The model to evaluate
        dataloader: DataLoader containing the dataset
        max_digit: Maximum digit value (default 9 for [0..9], use 4 for [0..4])
    """
    instance_risks = []
    
    for data in dataloader:
        images, labels, concepts = data
        images, labels, concepts = (
            images.to(model.device),
            labels.to(model.device),
            concepts.to(model.device),
        )
        
        out_dict = model(images)
        probs = torch.softmax(out_dict["pCS"], dim=-1) # shape: (Batch, 2, 10) or (Batch, 2, 5)
        
        # compute Probability of Satisfaction for each sample
        batch_risks = []
        for i in range(len(labels)):
            s = labels[i].item()
            
            p1 = probs[i, 0] # Prob vector for image 1
            p2 = probs[i, 1] # Prob vector for image 2
            
            # Sum probability of all VALID pairs (y1, y2) such that y1+y2 = s
            prob_valid = 0.0
            valid_pairs = get_valid_pairs(s, max_digit) # e.g. for s=1, max_digit=4 returns [(0,1), (1,0)]
            
            for (y1, y2) in valid_pairs:
                prob_valid += p1[y1] * p2[y2]
            
            # Risk = Probability of Constraint Violation
            batch_risks.append(1.0 - prob_valid.item())
            
        instance_risks.extend(batch_risks)
        
    return np.array(instance_risks)