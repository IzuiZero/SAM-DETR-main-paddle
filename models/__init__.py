from .fast_detr import build


def build_model(args):
    return build(args)
