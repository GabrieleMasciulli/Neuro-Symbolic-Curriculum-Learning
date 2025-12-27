import importlib
from pathlib import Path


def get_all_models():
    """Return all model module names in this package.

    NOTE: Must be independent from the current working directory (e.g., notebooks).
    """

    package_dir = Path(__file__).resolve().parent
    models = []
    for path in package_dir.iterdir():
        if not path.is_file():
            continue
        if path.suffix != ".py":
            continue
        if path.name.startswith("__"):
            continue
        models.append(path.stem)
    return sorted(models)


names = {}
for model in get_all_models():
    mod = importlib.import_module("models." + model)
    class_name = {x.lower(): x for x in mod.__dir__()}[model.replace("_", "")]
    names[model] = getattr(mod, class_name)


def get_model(args, encoder, decoder, n_images, c_split):
    if args.model == "cext":
        return names[args.model](encoder, n_images=n_images, c_split=c_split)
    elif args.model in [
        "mnistdpl",
        "mnistsl",
        "mnistltn",
        "kanddpl",
        "kandltn",
        "kandpreprocess",
        "kandclip",
        "minikanddpl",
        "mnistpcbmdpl",
        "mnistpcbmsl",
        "mnistpcbmltn",
        "mnistclip",
        "sddoiadpl",
        "sddoiacbm",
        "sddoialtn",
        "presddoiadpl",
        "boiadpl",
        "mnistcbm",
        "boiacbm",
        "boialtn",
        "kandcbm",
        "mnistnn",
        "kandnn",
        "sddoiann",
        "sddoiaclip",
        "boiann",
        "xorcbm",
        "xornn",
        "xordpl",
        "mnmathnn",
        "mnmathcbm",
        "mnmathdpl"
    ]:
        return names[args.model](
            encoder, n_images=n_images, c_split=c_split, args=args
        )  # only discriminative
    else:
        return names[args.model](
            encoder, decoder, n_images=n_images, c_split=c_split, args=args
        )
