from argparse import Namespace
from datasets.utils.base_dataset import BaseDataset, BOIA_get_loader
from datasets.utils.boia_creation import BOIADataset
from datasets.utils.sddoia_creation import CONCEPTS_ORDER
from backbones.boia_linear import BOIAConceptizer
from backbones.boia_mlp import BOIAMLP
from utils.risk_curriculum_sampler import (
    MultilabelClassSpecificRiskCurriculumSampler,
    MultilabelInstanceRiskCurriculumSampler,
)
import numpy as np
import os
import time


class BOIA(BaseDataset):
    NAME = "boia"

    def __init__(self, args: Namespace) -> None:
        super().__init__(args)

    def get_data_loaders(self):
        start = time.time()

        # Get the directory where this file is located, then navigate to data folder
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        data_dir = os.path.join(base_dir, "data", "bdd2048")

        image_dir = data_dir + "/"
        train_data_path = os.path.join(data_dir, "train_BDD_OIA.pkl")
        val_data_path = os.path.join(data_dir, "val_BDD_OIA.pkl")
        test_data_path = os.path.join(data_dir, "test_BDD_OIA.pkl")

        self.dataset_train = BOIADataset(
            pkl_file_path=train_data_path,
            use_attr=True,
            no_img=False,
            uncertain_label=False,
            image_dir=image_dir + "train",
            n_class_attr=2,
            transform=None,
            c_sup=self.args.c_sup,
            which_c=self.args.which_c,
            args=self.args,
        )
        self.dataset_val = BOIADataset(
            pkl_file_path=val_data_path,
            use_attr=True,
            no_img=False,
            uncertain_label=False,
            image_dir=image_dir + "val",
            n_class_attr=2,
            transform=None,
            args=self.args,
        )
        self.dataset_test = BOIADataset(
            pkl_file_path=test_data_path,
            use_attr=True,
            no_img=False,
            uncertain_label=False,
            image_dir=image_dir + "test",
            n_class_attr=2,
            transform=None,
            args=self.args,
        )

        print(f"Loaded datasets in {time.time()-start} s.")

        print(
            "Len loaders: \n train:",
            len(self.dataset_train),
            "\n val:",
            len(self.dataset_val),
        )
        print(" len test:", len(self.dataset_test))

        # ---- Curriculum learning setup ----
        self.curriculum_sampler = None
        self.concept_risks = None
        train_sampler = None

        if self.args.curriculum and self.args.risk_type:
            if self.args.risk_type == "class":
                risk_path = (
                    f"class_specific_risks_{self.args.dataset}_{self.args.model}.npy"
                )
                if os.path.exists(risk_path):
                    self.concept_risks = np.load(risk_path)
                    print(f"\n--- Curriculum Learning Enabled (multilabel/class) ---")
                    print(f"Loaded concept risks from {risk_path}")
                    print(f"Concept Risks: {self.concept_risks}\n")
                    self.curriculum_sampler = (
                        MultilabelClassSpecificRiskCurriculumSampler(
                            self.dataset_train,
                            self.concept_risks,
                            current_phase=1.0,
                        )
                    )
                    train_sampler = self.curriculum_sampler
                else:
                    print(
                        f"\nWarning: Curriculum enabled but risk file {risk_path} not found. "
                        "Using default sampler."
                    )

            elif self.args.risk_type == "instance":
                risk_path = (
                    f"instance_specific_risks_{self.args.dataset}_{self.args.model}.npy"
                )
                if os.path.exists(risk_path):
                    instance_risks = np.load(risk_path)
                    self.instance_risks = instance_risks
                    print(
                        f"\n--- Instance-Specific Curriculum Learning Enabled (multilabel) ---"
                    )
                    print(f"Loaded instance risks from {risk_path}")
                    self.curriculum_sampler = MultilabelInstanceRiskCurriculumSampler(
                        self.dataset_train,
                        self.instance_risks,
                        current_phase=1.0,
                    )
                    train_sampler = self.curriculum_sampler
                else:
                    print(
                        f"\nWarning: Curriculum enabled but risk file {risk_path} not found. "
                        "Using default sampler."
                    )

        self.train_loader = BOIA_get_loader(
            self.dataset_train,
            self.args.batch_size,
            val_test=False,
            sampler=train_sampler,
        )
        self.val_loader = BOIA_get_loader(
            self.dataset_val, self.args.batch_size, val_test=True
        )
        self.test_loader = BOIA_get_loader(
            self.dataset_test, self.args.batch_size, val_test=True
        )

        return self.train_loader, self.val_loader, self.test_loader

    def get_backbone(self):
        if self.args.backbone == "neural":
            return BOIAMLP(), None

        return BOIAConceptizer(din=2048, nconcept=21), None

    def get_split(self):
        return 1, ()

    def get_concept_labels(self):
        sorted_concepts = sorted(CONCEPTS_ORDER, key=CONCEPTS_ORDER.get)
        return sorted_concepts

    def give_supervision_to(self, indices):
        """Give concept supervision only to specific sample indices.

        Sets all training concepts to -1 (no supervision), then restores
        the real concept labels only for the specified indices.

        Args:
            indices: list or array of sample indices to supervise.
        """
        indices = np.array(indices, dtype=int)

        # Remove all supervision
        self.dataset_train.concepts[:] = -1

        # Restore supervision for selected indices
        self.dataset_train.concepts[indices] = self.dataset_train.real_concepts[indices]

        print(
            f"Supervision given to {len(indices)} / {len(self.dataset_train)} samples"
        )
        return self

    def print_stats(self):
        print("## Statistics ##")
        print("Train samples", len(self.dataset_train))
        print("Validation samples", len(self.dataset_val))
        print("Test samples", len(self.dataset_test))
