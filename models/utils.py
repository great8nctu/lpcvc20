import tensorflow as tf

from tensorflow.keras.models import model_from_json
from tensorflow.keras.layers import SeparableConv2D, Conv2D, Dense, BatchNormalization, DepthwiseConv2D

# Modified from https://github.com/triagemd/keras-trainer/blob/91269af898/keras_trainer/regularizations.py
def set_model_regularization(model, regularization_function, layers=None, bias_term=False, bn_term=True, depthwise_term=True):
    """
    Go through every Convolutional, Dense and BN layer in the model and apply the specified regularization function
    Args:
        model: Keras Machine Learning model
        regularization_function: Regularization function to include in layers
        layers: Layers to which the regularization will be applied
        bias_term: If true will add the regularization in the layer bias regularizer
    Returns: Model with regularization function applied
    """
    # Save weights in case it is a pre-trained model to not lose them afterwards
    model.save_weights("/tmp/tmp.h5")

    if layers is None:
        layers = model.layers

    for layer_index, layer in enumerate(layers):
        if isinstance(layer, SeparableConv2D):
            if depthwise_term:
                layer.depthwise_regularizer, layer.pointwise_regularizer = regularization_function, regularization_function
            if bias_term:
                layer.bias_regularizer = regularization_function
        elif isinstance(layer, DepthwiseConv2D):
            if depthwise_term:
                layer.depthwise_regularizer = regularization_function
            if bias_term:
                layer.bias_regularizer = regularization_function
        elif isinstance(layer, Conv2D) or isinstance(layer, Dense):
            layer.kernel_regularizer = regularization_function
            if bias_term:
                layer.bias_regularizer = regularization_function
        elif isinstance(layer, BatchNormalization):
            if bn_term:
                layer.gamma_regularizer, layer.beta_regularizer = regularization_function, regularization_function

    # This workaround is needed to add the regularizations and recover the original weights
    model = model_from_json(model.to_json())
    model.load_weights("/tmp/tmp.h5", by_name=True)
    return model
