import numpy as np
import torch
from torch.utils.data import Sampler, DataLoader


class ClassSpecificRiskCurriculumSampler(Sampler):
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
        all_concepts = dataset.real_concepts  # shape: (num_samples, 2)

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
            # score = (r1 + r2) / 2.0

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


class InstanceRiskCurriculumSampler(Sampler):
    def __init__(self, dataset, instance_risks, current_phase=1.0):
        """
        Args:
            dataset: The PyTorch dataset.
            instance_risks: A 1D numpy array or list of scores, one for each sample
                            in the dataset. (Low score = Easy, High score = Hard).
            current_phase: Float (0.0 to 1.0). Percentage of data to use.
        """
        self.dataset = dataset
        self.current_phase = current_phase

        if torch.is_tensor(instance_risks):
            self.instance_risks = instance_risks.cpu().numpy()
        else:
            self.instance_risks = np.array(instance_risks)

        # sort indices by difficulty (Low Risk -> High Risk)
        self.sorted_indices = np.argsort(self.instance_risks)

    def __iter__(self):
        # determine how many samples are allowed in this phase
        num_samples = len(self.dataset)
        cutoff = int(num_samples * self.current_phase)
        cutoff = max(cutoff, 64)

        # select the easiest 'cutoff' samples
        active_indices = self.sorted_indices[:cutoff]

        np.random.shuffle(active_indices)

        return iter(active_indices)

    def __len__(self):
        cutoff = int(len(self.dataset) * self.current_phase)
        return max(cutoff, 64)


class MultilabelClassSpecificRiskCurriculumSampler(Sampler):
    """Curriculum sampler for multilabel datasets (e.g. BOIA).

    Each sample has multiple binary concept attributes. The per-concept risk
    scores are combined into a sample difficulty via the **mean** risk of all
    concepts that are active (=1) for that sample.  If no concept is active the
    difficulty falls back to the overall mean risk.
    """

    def __init__(self, dataset, risk_scores, current_phase=1.0):
        """
        Args:
            dataset: PyTorch dataset with a ``real_concepts`` attribute of
                     shape (num_samples, num_concepts) containing 0/1 values.
            risk_scores: 1-D array of per-concept risk scores [R_0, …, R_C].
            current_phase: Float (0.0 to 1.0). Fraction of data to use.
        """
        self.dataset = dataset
        self.risk_scores = np.asarray(risk_scores, dtype=float)
        self.current_phase = current_phase

        all_concepts = dataset.real_concepts  # shape: (N, C)
        if torch.is_tensor(all_concepts):
            all_concepts = all_concepts.cpu().numpy()

        self.sample_difficulties = self._score_samples(all_concepts)
        self.sorted_indices = np.argsort(self.sample_difficulties)

    def _score_samples(self, concepts):
        """Average risk across active concepts for each sample."""
        difficulties = []
        mean_risk = float(np.mean(self.risk_scores))
        for row in concepts:
            active_mask = row.astype(bool)
            if active_mask.any():
                score = float(np.mean(self.risk_scores[active_mask]))
            else:
                score = mean_risk
            difficulties.append(score)
        return np.array(difficulties)

    def __iter__(self):
        num_samples = len(self.dataset)
        cutoff = int(num_samples * self.current_phase)
        cutoff = max(cutoff, 64)
        active_indices = self.sorted_indices[:cutoff].copy()
        np.random.shuffle(active_indices)
        return iter(active_indices)

    def __len__(self):
        cutoff = int(len(self.dataset) * self.current_phase)
        return max(cutoff, 64)


class MultilabelInstanceRiskCurriculumSampler(Sampler):
    """Instance-level curriculum sampler for multilabel datasets (e.g. BOIA).

    Uses pre-computed per-sample risk scores (average concept prediction
    error) to order samples from easy to hard.
    """

    def __init__(self, dataset, instance_risks, current_phase=1.0):
        """
        Args:
            dataset: The PyTorch dataset.
            instance_risks: 1-D array of per-sample risk scores.
            current_phase: Float (0.0 to 1.0). Fraction of data to use.
        """
        self.dataset = dataset
        self.current_phase = current_phase

        if torch.is_tensor(instance_risks):
            self.instance_risks = instance_risks.cpu().numpy()
        else:
            self.instance_risks = np.array(instance_risks)

        self.sorted_indices = np.argsort(self.instance_risks)

    def __iter__(self):
        num_samples = len(self.dataset)
        cutoff = int(num_samples * self.current_phase)
        cutoff = max(cutoff, 64)
        active_indices = self.sorted_indices[:cutoff].copy()
        np.random.shuffle(active_indices)
        return iter(active_indices)

    def __len__(self):
        cutoff = int(len(self.dataset) * self.current_phase)
        return max(cutoff, 64)
