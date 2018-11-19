import json
import os
import sys

from sacred import Experiment
from sacred.observers import FileStorageObserver

from constants import *
sys.path.append(BASE_DIR)
from shield.attack import attack
from shield.evaluate import evaluate
import shield.opts as opts
from shield.preprocess import preprocess


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

ex = Experiment('shield-experiment')
ex.observers.append(FileStorageObserver.create(os.path.join(RUNS_DIR)))


@ex.config
def default_config():
    use_gpu = None
    perform = None
    model, model_checkpoint_path = None, None
    attack, attack_options = None, None

    if attack_options is not None:
        attack_options = json.loads(str(attack_options))


@ex.main
def main(use_gpu,
         perform,
         model, model_checkpoint_path,
         attack, attack_options,
         _run, _log):
    run_id = os.path.basename(_run.observers[0].dir)

    if use_gpu is not None:
        print('Using GPU(s) %s' % use_gpu)
        os.environ['CUDA_VISIBLE_DEVICES'] = str(use_gpu)

    assert perform in ['attack', 'preprocess', 'evaluate']

    if perform == 'attack':
        assert model in opts.models
        assert attack in opts.attacks

        attack_options_ = opts.attack_options[attack].copy()
        if attack_options is not None:
            if attack == 'fgsm' and 'eps' in attack_options:
                attack_options['eps'] = 2. * attack_options['eps'] / 255.
            attack_options_.update(attack_options)
        attack_identifier = opts.attack_identifiers[attack](attack_options_)

        experiment_identifier = '-'.join([
            'imagenet_val', model, '%s_%s' % (attack, attack_identifier)])

        output_dir = \
            os.path.join(ATTACKED_OUT_DIR, experiment_identifier, run_id)
        assert not os.path.exists(output_dir)
        os.makedirs(output_dir)


if __name__ == '__main__':
    ex.run_commandline()
