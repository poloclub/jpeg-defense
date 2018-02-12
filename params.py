import numpy as np
from cleverhans.attacks import FastGradientMethod, DeepFool, CarliniWagnerL2

from constants import CHECKPOINTS_DIR
from models.inception_v4 import InceptionV4
from models.resnet_50_v2 import ResNet50v2
from models.resnet_101_v2 import ResNet101v2
from utils.defenses import jpeg_compress, slq_tf, denoise_median_filter, denoise_tv_bregman


# -----

models = ['inception_v4', 'resnet_50_v2', 'resnet_101_v2']
attacks = ['fgsm', 'df', 'cwl2']
defenses = ['jpeg', 'median_filter', 'tv_bregman', 'nl_means']

# -----

model_class_map = {
    'inception_v4': InceptionV4,
    'resnet_50_v2': ResNet50v2,
    'resnet_101_v2': ResNet101v2
}

model_checkpoint_map = {
    'inception_v4': CHECKPOINTS_DIR+'inception_v4.ckpt',
    'resnet_50_v2': CHECKPOINTS_DIR+'resnet_50_v2.ckpt',
    'resnet_101_v2': CHECKPOINTS_DIR+'resnet_101_v2.ckpt'
}

model_slim_name_map = {
    'inception_v4': 'inception_v4',
    'resnet_50_v2': 'resnet_v2_50',
    'resnet_101_v2': 'resnet_v2_101'
}

# -----

attack_class_map = {
    'fgsm': FastGradientMethod,
    'df': DeepFool,
    'cwl2': CarliniWagnerL2
}

attack_ablations = {
    'fgsm': [
        {'ord': np.inf, 'eps': (2. * eps / 255.), 'clip_min': -1., 'clip_max': 1.}
        for eps in range(2, 17, 2)
    ],
    'df': [{'nb_candidate': 10, 'max_iter': 100, 'clip_min': -1., 'clip_max': 1.}],
    'cwl2': [
        {'confidence': 0, 'learning_rate': 5e-3, 'batch_size': 4, 'clip_min': -1., 'clip_max': 1.}]
}

attack_identifiers = {
    'fgsm': lambda p: 'ord_%s-eps_%s' % (p['ord'], int(p['eps'] * 255 / 2),),
    'df': lambda p: 'noparams',
    'cwl2': lambda p: 'conf_%s' % p['confidence']
}

# -----

defense_fn_map = {
    'jpeg': jpeg_compress,
    'median_filter': denoise_median_filter,
    'tv_bregman': denoise_tv_bregman,
    'slq': slq_tf
}

defense_ablations = {
    'jpeg': [{'quality': q} for q in range(100, 19, -10)],
    'median_filter': [{'size': k} for k in [3, 5]],
    'tv_bregman': [{'weight': w} for w in [30]],
    'slq': [{'run': r} for r in range(10)]
}

defense_identifiers = {
    'jpeg': lambda p: 'qual_%s' % (p['quality'],),
    'median_filter': lambda p: 'size_%s' % (p['size'],),
    'tv_bregman': lambda p: 'weight_%s' % (p['weight'],),
    'slq': lambda p: 'run_%s' % (p['run'],)
}
