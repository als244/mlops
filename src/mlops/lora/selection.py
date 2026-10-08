"""One-time conversion and explicit trainability, outside model execution."""

from fnmatch import fnmatchcase

from torch import nn

from ..modules import LanguageModelHead
from .linear import LoRAHead, LoRALinear


def apply_lora(model, config, *, targets, trainable_base=(), converters=None):
    """Replace selected modules in place, freezing all unspecified base parameters.

    targets and trainable_base are module/parameter path globs, respectively.
    converters optionally maps module classes to (module, config) factories.
    All aliases retain one replacement and shared Parameter identities.
    Returns the model, or a replacement when the root module is selected.
    """
    targets, trainable_base = tuple(targets), tuple(trainable_base)
    if not targets:
        raise ValueError("targets must not be empty")
    if any(hasattr(module, "lora_config") for module in model.modules()):
        raise ValueError("model already contains LoRA modules")
    modules = dict(model.named_modules(remove_duplicate=False))
    parameters = dict(model.named_parameters(remove_duplicate=False))
    matched = {}
    for pattern in targets:
        names = [name for name in modules if fnmatchcase(name, pattern)]
        if not names:
            raise ValueError(f"target {pattern!r} matched no modules")
        matched.update((name, modules[name]) for name in names)
    for name in matched:
        if any(
            parent != name and (not parent or name.startswith(parent + "."))
            for parent in matched
        ):
            raise ValueError("select a module or its children, not both")
    trainable_ids = set()
    for pattern in trainable_base:
        names = [name for name in parameters if fnmatchcase(name, pattern)]
        if not names:
            raise ValueError(f"trainable_base {pattern!r} matched no parameters")
        trainable_ids.update(id(parameters[name]) for name in names)
    factories = {
        nn.Linear: LoRALinear,
        LanguageModelHead: LoRAHead,
        **(converters or {}),
    }
    selected = {}
    for name, module in matched.items():
        factory = next(
            (factories[t] for t in type(module).__mro__ if t in factories), None
        )
        if factory is None:
            raise TypeError(f"no LoRA conversion for {name!r}: {type(module).__name__}")
        selected[id(module)] = module, factory
    replacements = {
        identity: factory(module, config)
        for identity, (module, factory) in selected.items()
    }
    for parameter in parameters.values():
        parameter.requires_grad_(id(parameter) in trainable_ids)
    for name, module in modules.items():
        if id(module) not in replacements:
            continue
        replacement = replacements[id(module)]
        if not name:
            model = replacement
        else:
            parent, _, leaf = name.rpartition(".")
            setattr(modules[parent], leaf, replacement)
    return model


def parameter_report(model):
    """Counts and dtype/shape details for the parameters an optimizer will update."""
    rows = [
        {
            "name": name,
            "shape": list(p.shape),
            "elements": p.numel(),
            "dtype": str(p.dtype),
            "bytes": p.numel() * p.element_size(),
        }
        for name, p in model.named_parameters()
        if p.requires_grad
    ]
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(row["elements"] for row in rows)
    return {
        "total_parameters": total,
        "trainable_parameters": trainable,
        "frozen_parameters": total - trainable,
        "trainable": rows,
    }
