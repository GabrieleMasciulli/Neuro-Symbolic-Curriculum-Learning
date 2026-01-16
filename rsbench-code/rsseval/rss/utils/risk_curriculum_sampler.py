import numpy as np
import torch
from torch.utils.data import Sampler, DataLoader

class RiskCurriculumSampler(Sampler):
    def __init__(self, dataset, risk_scores, current_phase=1.0):
        """
        Args:
            dataset: PyTorch dataset where dataset[i] returns (data, label, concepts)
                     concepts should be the hidden ground truth (c1, c2).
            concept_risks: Array of risks [R_0, R_1, ..., R_9]
            current_phase: Float (0.0 to 1.0). Percentage of data to use.
                           1.0 = All data.
        """
        self.dataset = dataset
        self.risk_scores = risk_scores
        self.current_phase = current_phase

        # assign difficulty scores to each sample based on its concepts
        all_concepts = dataset.real_concepts # shape: (num_samples, 2)
        
        # Ensure concepts are numpy/cpu
        if torch.is_tensor(all_concepts):
            all_concepts = all_concepts.cpu().numpy()    
        
        self.sample_difficulties = self._score_samples(all_concepts)
        
        # Sort indices by difficulty (Low -> High)
        self.sorted_indices = np.argsort(self.sample_difficulties)
        
    def _score_samples(self, concepts):
        """
        Combines concept risks into a sample difficulty score.
        Strategy: MAX risk among the concepts present in the sample.
        """
        difficulties = []
        for pair in concepts:
            c1, c2 = pair[0], pair[1]
            
            r1 = self.risk_scores[c1]
            r2 = self.risk_scores[c2]
            
            # Difficulty = Max(Risk(c1), Risk(c2))
            score = max(r1, r2)
            #score = (r1 + r2) / 2.0  
            
            difficulties.append(score)
        return np.array(difficulties)
    
    def __iter__(self):
        # select the top N% easiest samples
        num_samples = len(self.dataset)
        cutoff = int(num_samples * self.current_phase)
        
        # easiest subset
        active_indices = self.sorted_indices[:cutoff]
        
        np.random.shuffle(active_indices)
        
        return iter(active_indices)

    def __len__(self):
        return int(len(self.dataset) * self.current_phase)