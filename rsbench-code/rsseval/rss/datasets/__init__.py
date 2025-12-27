import inspect
import importlib
from pathlib import Path
from argparse import Namespace


def get_all_datasets():
    """Return all dataset module names in this package.

    NOTE: Must be independent from the current working directory (e.g., notebooks).
    """

    package_dir = Path(__file__).resolve().parent
    datasets = []
    for path in package_dir.iterdir():
        if not path.is_file():
            continue
        if path.suffix != ".py":
            continue
        if path.name.startswith("__"):
            continue
        datasets.append(path.stem)
    return sorted(datasets)


NAMES = {}
for dataset in get_all_datasets():
    dat = importlib.import_module("datasets." + dataset)

    dataset_classes_name = [
        x
        for x in dat.__dir__()
        if "type" in str(type(getattr(dat, x)))
        and "BaseDataset" in str(inspect.getmro(getattr(dat, x))[1:])
    ]
    for d in dataset_classes_name:
        c = getattr(dat, d)
        NAMES[c.NAME] = c


def get_dataset(args: Namespace):
    """
    Creates and returns a continual dataset.
    :param args: the arguments which contains the hyperparameters
    :return: the continual dataset
    """
    assert args.dataset in NAMES.keys(), f"{args.dataset} in {NAMES}"
    return NAMES[args.dataset](args)
