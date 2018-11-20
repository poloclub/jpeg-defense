import os


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
OUT_DIR = os.path.join(BASE_DIR, 'out')

CHECKPOINTS_DIR = os.path.join(DATA_DIR, 'checkpoints')
VALIDATION_DATA_DIR = os.path.join(DATA_DIR, 'validation')

RUNS_DIR = os.path.join(OUT_DIR, 'runs')
ATTACKED_OUT_DIR = os.path.join(OUT_DIR, 'attacked')
PREPROCESSED_OUT_DIR = os.path.join(OUT_DIR, 'preprocessed')

ATTACKED_TFRECORD_FILENAME = 'attacked.tfrecord'
PREPROCESSED_TFRECORD_FILENAME = 'preprocessed.tfrecord'
ACCURACY_NPZ_FILENAME = 'accuracy.npz'
TOP5_ACCURACY_NPZ_FILENAME = 'top5_accuracy.npz'
NORMALIZED_L2_DISTANCE_NPZ_FILENAME = 'normalized_l2_distance.npz'

NUM_SAMPLES_VALIDATIONSET = 50000
RESNET_IMAGE_SIZE = 299

if __name__ == '__main__':
    print(BASE_DIR)
