from models import encoder
from models import losses
from models import ssl
from .resnet import ResNet

REGISTERED_MODELS = {
    "sim-clr": ssl.SimCLR,
    "eval": ssl.SSLEval,
    "semi-supervised-eval": ssl.SemiSupervisedEval,
}
