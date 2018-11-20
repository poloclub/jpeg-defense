import glob
import json
import os
import sys

import warnings
warnings.filterwarnings('ignore')

from sacred import Experiment
from sacred.observers import FileStorageObserver
import tensorflow as tf

from constants import *
sys.path.append(BASE_DIR)
from shield.attack import attack as shield_attack
from shield.evaluate import evaluate as shield_evaluate
import shield.opts as opts
from shield.preprocess import preprocess as shield_preprocess


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
tf.logging.set_verbosity(tf.logging.ERROR)

ex = Experiment('shield-experiment')
ex.observers.append(FileStorageObserver.create(os.path.join(RUNS_DIR)))


@ex.config
def default_config():
    use_gpu = None
    perform = None
    model, model_checkpoint_path = None, None
    attack, attack_options = None, {}
    defense, defense_options = None, {}


@ex.main
def main(use_gpu, perform,
         model, model_checkpoint_path,
         attack, attack_options,
         defense, defense_options,
         _run, _log):

    def _get_file_from_latest_run(directory, filename):
        runs = glob.glob('%s-*' % os.path.join(directory, 'run'))
        runs = map(lambda p: os.path.basename(p), runs)
        last_run = max(runs, key=lambda n: int(n.split('-')[-1]))
        return os.path.join(directory, last_run, filename)

    assert perform in ['attack', 'defend', 'evaluate']
    assert model in opts.models
    assert (attack in opts.attacks if attack is not None else True)
    assert (defense in opts.defenses if defense is not None else True)

    experiment_id = ['imagenet_val', model]
    run_id = os.path.basename(_run.observers[0].dir)

    if use_gpu is not None:
        print('Using GPU(s) %s' % use_gpu)
        os.environ['CUDA_VISIBLE_DEVICES'] = str(use_gpu)

    if attack == 'fgsm' and 'eps' in attack_options:
        attack_options['eps'] = 2. * attack_options['eps'] / 255.
    attack_options_updated = opts.attack_options[attack].copy()
    attack_options_updated.update(attack_options)

    defense_options_updated = None
    if defense is not None:
        defense_options_updated = opts.defense_options[defense].copy()
        defense_options_updated.update(defense_options)

    if perform == 'attack':
        assert attack is not None

        input_tfrecords = os.path.join(VALIDATION_DATA_DIR, '*')

        attack_identifier = \
            opts.attack_identifiers[attack](attack_options_updated)
        experiment_id.append('%s_%s' % (attack, attack_identifier))
        experiment_id = '-'.join(experiment_id)

        output_dir = os.path.join(
            ATTACKED_OUT_DIR, experiment_id, 'run-%s' % run_id)
        assert not os.path.exists(output_dir)
        os.makedirs(output_dir)

        shield_attack(
            input_tfrecords, model,
            attack, attack_options_updated, output_dir,
            load_jpeg=True, decode_pixels=True,
            model_checkpoint_path=model_checkpoint_path)

    elif perform == 'defend':
        assert attack is not None
        assert defense is not None

        attack_identifier = \
            opts.attack_identifiers[attack](attack_options_updated)
        experiment_id.append('%s_%s' % (attack, attack_identifier))

        attack_experiment_id = '-'.join(experiment_id)
        attack_dir = os.path.join(ATTACKED_OUT_DIR, attack_experiment_id)
        assert os.path.exists(attack_dir), \
            'Attack needs to be performed first'
        input_tfrecords = _get_file_from_latest_run(
            attack_dir, ATTACKED_TFRECORD_FILENAME)

        defense_identifier = \
            opts.defense_identifiers[defense](defense_options_updated)
        experiment_id.append('%s_%s' % (defense, defense_identifier))
        experiment_id = '-'.join(experiment_id)

        output_dir = os.path.join(
            PREPROCESSED_OUT_DIR, experiment_id, 'run-%s' % run_id)
        assert not os.path.exists(output_dir)
        os.makedirs(output_dir)

        shield_preprocess(
            input_tfrecords,
            defense, defense_options_updated,
            output_dir,
            image_size=RESNET_IMAGE_SIZE,
            load_jpeg=False, decode_pixels=False)

    elif perform == 'evaluate':
        assert attack is not None

        attack_identifier = \
            opts.attack_identifiers[attack](attack_options_updated)
        experiment_id.append('%s_%s' % (attack, attack_identifier))

        input_tfrecords = None
        load_jpeg, decode_pixels = None, None

        if defense is None:
            attack_experiment_id = '-'.join(experiment_id)
            attack_dir = os.path.join(ATTACKED_OUT_DIR, attack_experiment_id)
            assert os.path.exists(attack_dir), \
                'Attack needs to be performed first'
            input_tfrecords = _get_file_from_latest_run(
                attack_dir, ATTACKED_TFRECORD_FILENAME)
            load_jpeg, decode_pixels = False, False
        else:
            defense_identifier = \
                opts.defense_identifiers[defense](defense_options_updated)
            experiment_id.append('%s_%s' % (defense, defense_identifier))
            defense_experiment_id = '-'.join(experiment_id)
            defense_dir = os.path.join(
                PREPROCESSED_OUT_DIR, defense_experiment_id)
            assert os.path.exists(defense_dir), \
                'Defense needs to be performed first'
            input_tfrecords = _get_file_from_latest_run(
                defense_dir, PREPROCESSED_TFRECORD_FILENAME)
            load_jpeg, decode_pixels = False, True

        output_dir = os.path.dirname(input_tfrecords)
        assert os.path.exists(output_dir)

        shield_evaluate(
            input_tfrecords,
            model, output_dir,
            model_checkpoint_path=model_checkpoint_path,
            load_jpeg=load_jpeg, decode_pixels=decode_pixels)


if __name__ == '__main__':
    ex.run_commandline()
