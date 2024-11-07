from .darknet19 import build_darknet19


def build_backbone(pretrained=True):
    return build_darknet19(pretrained).to('cuda')
