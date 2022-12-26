from models import encoder
from models import losses
from models import resnet
from models import ssl
from models import trainernet

REGISTERED_MODELS = {
    'sim-clr': ssl.SimCLR,
    'eval': ssl.SSLEval,
    'semi-supervised-eval': ssl.SemiSupervisedEval,
}
