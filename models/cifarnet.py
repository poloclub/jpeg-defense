import numpy as np

from keras.layers import Input, Dense, Dropout, Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.models import Model as KerasModel, Sequential

from cleverhans.utils_keras import KerasModelWrapper

from heimdall.model import BaseModel as HeimdallModel


def _build_cifarnet_model(weights_file=None):
    inputs = Input(
        shape=(32, 32, 3,),
        name='input')

    conv1 = Conv2D(
        32, (3, 3),
        padding='same',
        activation='relu',
        name='conv1')(inputs)

    conv2 = Conv2D(
        32, (3, 3),
        padding='valid',
        activation='relu',
        name='conv2')(conv1)

    pool1 = MaxPooling2D(
        pool_size=(3, 3,),
        strides=(2, 2,),
        padding='same',
        name='pool1')(conv2)

    drop1 = Dropout(0.25)(pool1)

    conv3 = Conv2D(
        64, (3, 3),
        padding='same',
        activation='relu',
        name='conv3')(drop1)

    conv4 = Conv2D(
        64, (3, 3),
        padding='valid',
        activation='relu',
        name='conv4')(conv3)

    pool2 = MaxPooling2D(
        pool_size=(3, 3,),
        strides=(2, 2,),
        padding='same',
        name='pool2')(conv4)

    drop2 = Dropout(0.25)(pool2)

    fcl1 = Dense(
        512,
        activation='relu',
        name='fc1')(Flatten()(drop2))

    drop3 = Dropout(0.5)(fcl1)

    preds = Dense(
        10,
        activation='softmax',
        name='preds')(drop3)

    keras_model = KerasModel(inputs=inputs, outputs=preds)
    
    if weights_file is not None:
        keras_model.load_weights(weights_file)
    
    return keras_model


class CifarNet(KerasModelWrapper, HeimdallModel):
    def __init__(self, weights_file=None):
        super(CifarNet, self).__init__(
            model=_build_cifarnet_model(weights_file))
        
        self.set_batch_preprocessor(lambda X: X.astype('float32') / 255.)
        
    def predict_labels(self, batch=None):
        return np.argmax(self.model.predict(batch), axis=1)
