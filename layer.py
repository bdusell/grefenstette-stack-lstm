import dynet

class ActivationFunction:

    def __init__(self, function, glorot_gain):
        self.function = function
        self.glorot_gain = glorot_gain

    def __call__(self, arg):
        return self.function(arg)

tanh = ActivationFunction(dynet.tanh, 1.0)
relu = ActivationFunction(dynet.rectify, 0.5)
sigmoid = ActivationFunction(dynet.logistic, 4.0)
linear = ActivationFunction(lambda x: x, 1.0)

class LayerInterface:

    def expression(self):
        raise NotImplementedError

class LayerExpressionInterface:

    def __call__(self, arg):
        raise NotImplementedError

class Layer(LayerInterface):

    def __init__(self, weights, bias, activation_function):
        self.weights = weights
        self.bias = bias
        self.activation_function = activation_function

    def expression(self):
        return LayerExpression(
            dynet.parameter(self.weights),
            dynet.parameter(self.bias),
            self.activation_function)

    @property
    def input_size(self):
        return self.weights.shape()[1]

    @property
    def output_size(self):
        return self.weights.shape()[0]

class LayerExpression(LayerExpressionInterface):

    def __init__(self, weights, bias, activation_function):
        self.weights = weights
        self.bias = bias
        self.activation_function = activation_function

    def __call__(self, arg):
        return self.activation_function(dynet.affine_transform([self.bias, self.weights, arg]))

class CompoundLayer(LayerInterface):

    def __init__(self, layers):
        self.layers = layers

    def expression(self):
        return CompoundLayerExpression([l.expression() for l in self.layers])

class CompoundLayerExpression(LayerExpressionInterface):

    def __init__(self, layers):
        self.layers = layers

    def __call__(self, arg):
        result = arg
        for layer in self.layers:
            result = layer(result)
        return result

def add_layer(params, input_size, output_size, activation_function,
        weights_initializer=None, bias_initializer=None, name=None):
    params = params.add_subcollection(name)
    if weights_initializer is None:
        weights_initializer = dynet.GlorotInitializer(
            False, activation_function.glorot_gain)
    if bias_initializer is None:
        bias_initializer = dynet.GlorotInitializer(
            False, activation_function.glorot_gain)
    return Layer(
        params.add_parameters((output_size, input_size), weights_initializer,
            b'weights'),
        params.add_parameters(output_size, bias_initializer, b'bias'),
        activation_function)

def combine_layers(layers):
    return CompoundLayer(layers)
