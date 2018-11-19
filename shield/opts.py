import os

import numpy as np
from cleverhans.attacks import FastGradientMethod, DeepFool, CarliniWagnerL2

from shield.constants import CHECKPOINTS_DIR
from shield.models.resnet_50_v2 import ResNet50v2
from shield.utils.defenses import (jpeg_compress, slq,
                                   median_filter,
                                   denoise_tv_bregman)


# -----

models = ['resnet_50_v2']
attacks = ['fgsm', 'df', 'cwl2']
defenses = ['jpeg', 'slq', 'median_filter', 'tv_bregman']
tf_defenses = ['jpeg', 'slq']

# -----

model_class_map = {
    'resnet_50_v2': ResNet50v2}

model_checkpoint_map = {
    'resnet_50_v2': os.path.join(CHECKPOINTS_DIR, 'resnet_50_v2.ckpt')}

model_slim_name_map = {
    'resnet_50_v2': 'resnet_v2_50'}

# -----

attack_class_map = {
    'fgsm': FastGradientMethod,
    'df': DeepFool,
    'cwl2': CarliniWagnerL2}

attack_options = {
    'fgsm': {
        'ord': np.inf,
        'eps': (2. * 8. / 255.),
        'clip_min': -1., 'clip_max': 1.},
    'df': {
        'nb_candidate': 10,
        'max_iter': 100,
        'clip_min': -1., 'clip_max': 1.},
    'cwl2': {
        'confidence': 0,
        'learning_rate': 5e-3,
        'batch_size': 4,
        'clip_min': -1., 'clip_max': 1.}}

attack_identifiers = {
    'fgsm': lambda p: 'ord_%s_eps_%s' % (p['ord'], int(p['eps'] * 255 / 2)),
    'df': lambda p: 'noparams',
    'cwl2': lambda p: 'conf_%s' % p['confidence']}

# -----

defense_fn_map = {
    'jpeg': jpeg_compress,
    'slq': slq,
    'median_filter': median_filter,
    'tv_bregman': denoise_tv_bregman}

defense_options = {
    'jpeg': {'quality': 80},
    'slq': {'run': 1},
    'median_filter': {'size': 3},
    'tv_bregman': {'weight': 30}}

defense_identifiers = {
    'jpeg': lambda p: 'qual_%s' % p['quality'],
    'slq': lambda p: 'run_%s' % p['run'],
    'median_filter': lambda p: 'size_%s' % p['size'],
    'tv_bregman': lambda p: 'weight_%s' % p['weight']}
