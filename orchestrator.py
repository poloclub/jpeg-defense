import os
import argparse
import json

from constants import ADVERSARIAL_OUT_DIR, EVAL_OUT_DIR, PREPROCESSED_OUT_DIR, VALIDATIONSET_RECORDS_EXPRESSION
from attack import attack_and_save_images
from defend import defend_and_save_images
from evaluate import evaluate_record
from params import *


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'


argparser = argparse.ArgumentParser(description='Perform ImageNet experiments.')

argparser.add_argument('--perform', type=str,
                       choices=['attack', 'defend', 'evaluate'],
                       help='Action to be performed',
                       required=True)
argparser.add_argument('--use_gpu', type=str, help='GPU IDs, comma-separated', required=True)
argparser.add_argument('--debug', type=str, help='Debug mode', required=True)
argparser.add_argument('--models', type=str, help='Model names, comma-separated')
argparser.add_argument('--checkpoint_paths', type=str, help='Model checkpoint paths, comma-separated')
argparser.add_argument('--attacks', type=str, help='Attack names, comma-separated')
argparser.add_argument('--defenses', type=str, help='Defense names, comma-separated')
argparser.add_argument('--blackbox', type=str, help='Model name to be used for black-box evaluation')
argparser.add_argument('--attack_ablations', type=str, help='Attack ablations, json-format')
argparser.add_argument('--defense_ablations', type=str, help='Defense ablations, json-format')
argparser.add_argument('--shard_range', type=str, help='Shard range for CWL2 attack')
argparser.add_argument('--retrained_model_quality', type=str, help='JPEG quality model was retrained on')
argparser.add_argument('--experiment_scope', type=str, help='Experiment scope for getting notified')

args = argparser.parse_args()

args.debug = args.debug.lower() != 'false'

if not args.debug:
    confirm = 'y'

    if confirm != 'y':
        exit()
else:
    print "WARN: Debug mode is ON!"

print "Using GPU(s)", args.use_gpu
os.environ["CUDA_VISIBLE_DEVICES"] = args.use_gpu

models = args.models.split(',')
if args.checkpoint_paths is not None:
    checkpoint_paths = args.checkpoint_paths.split(',')

    assert len(checkpoint_paths) == len(models)

    for i, model in enumerate(models):
        model_checkpoint_map[model] = checkpoint_paths[i]


if args.attack_ablations is not None:
    attack_ablations = json.loads(args.attack_ablations)

    if 'fgsm' in attack_ablations:
        attack_ablations['fgsm'] = map(
            lambda abl: {'ord': abl['ord'], 'eps': (2. * abl['eps'] / 255.)},
            attack_ablations['fgsm'])

if args.defense_ablations is not None:
    defense_ablations = json.loads(args.defense_ablations)


if args.perform == 'attack':
    attacks = args.attacks.split(',')

    for model in models:
        for attack in attacks:
            for attack_ablation in attack_ablations[attack]:
                opt = {
                    'model': model,
                    'attack': attack,
                    'attack_params': attack_ablation,
                    'attack_identifier': attack_identifiers[attack](attack_ablation)
                }

                if attack == 'cwl2':
                    assert args.shard_range is not None
                    i, j = map(lambda _: int(_), args.shard_range.split(','))
                    opt['attack_identifier'] += '-'+('shards_%03d_%03d' % (i, j))
                    opt['attack_shards'] = (i, j,)

                print opt

                attack_and_save_images(opt, debug=args.debug, experiment_scope=args.experiment_scope)

elif args.perform == 'defend':
    attacks = args.attacks.split(',')
    defenses = args.defenses.split(',')

    for model in models:
        for defense in defenses:
            for attack in attacks:
                for defense_ablation in defense_ablations[defense]:
                    for attack_ablation in attack_ablations[attack]:
                        opt = {
                            'model': model,
                            'defense': defense,
                            'defense_params': defense_ablation,
                            'defense_identifier': defense_identifiers[defense](defense_ablation),
                            'attack': attack,
                            'attack_params': attack_ablation,
                            'attack_identifier': attack_identifiers[attack](attack_ablation)
                        }

                        print opt

                        defend_and_save_images(opt, debug=args.debug, experiment_scope=args.experiment_scope)

elif args.perform == 'evaluate':
    attacks = args.attacks.split(',') if args.attacks is not None else []
    defenses = args.defenses.split(',') if args.defenses is not None else []

    blackbox_model = args.blackbox

    if args.retrained_model_quality is not None:
        model_checkpoint_map['resnet_50_v2'] = retrained_model_checkpoints[args.retrained_model_quality]
        print 'loading model weights for ResNet50v2 from', model_checkpoint_map['resnet_50_v2']

        EVAL_OUT_DIR = OUT_DIR + 'eval_50k_5e-3_240k/eval_%s/' % args.retrained_model_quality
        print 'writing evaluation to', EVAL_OUT_DIR

    for model in models:
        if len(attacks) == 0:
            evaluate_record(
                VALIDATIONSET_RECORDS_EXPRESSION, model,
                model_checkpoint_path=model_checkpoint_map[model],
                load_jpeg=True, decode_pixels=True,
                save_eval_to_path=EVAL_OUT_DIR + '-'.join(['imagenet_val', model]) + '.npz',
                debug=args.debug, experiment_scope=args.experiment_scope)

        for attack in attacks:
            for attack_ablation in attack_ablations[attack]:
                attack_identifier = attack_identifiers[attack](attack_ablation)
                attack_tfrecord_fname_prefix = '-'.join(
                    [
                        'imagenet_val',
                        model if blackbox_model is None else blackbox_model,
                        attack, attack_identifier
                     ])

                attacked_images_path = ADVERSARIAL_OUT_DIR + attack_tfrecord_fname_prefix+'.tfrecord' \
                    if attack != 'cwl2' \
                    else ADVERSARIAL_OUT_DIR + 'imagenet_val-resnet_50_v2-cwl2-conf_0-shards_*'

                for defense in defenses:
                    for defense_ablation in defense_ablations[defense]:
                        defense_identifier = defense_identifiers[defense](defense_ablation)
                        defense_tfrecord_fname_prefix = '-'.join(
                            [attack_tfrecord_fname_prefix, defense, defense_identifier])

                        print 'evaluating', defense_tfrecord_fname_prefix, 'with', model
                        evaluate_record(
                            PREPROCESSED_OUT_DIR + defense_tfrecord_fname_prefix+'.tfrecord', model,
                            model_checkpoint_path=model_checkpoint_map[model],
                            load_jpeg=False, decode_pixels=True,
                            save_eval_to_path=EVAL_OUT_DIR \
                                              + defense_tfrecord_fname_prefix \
                                              + (('-blackbox-'+model) if blackbox_model is not None else '') \
                                              +'.npz',
                            debug=args.debug, experiment_scope=args.experiment_scope)
