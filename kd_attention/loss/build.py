from .catalog import Catalog


def build_loss(name, **kwargs):
    return Catalog.losses[name](**kwargs)
