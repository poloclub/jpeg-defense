import os


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
OUT_DIR = os.path.join(BASE_DIR, 'out')
CHECKPOINTS_DIR = os.path.join(DATA_DIR, 'checkpoints')

NUM_SAMPLES_VALIDATIONSET = 50000

if __name__ == '__main__':
    print(BASE_DIR)
