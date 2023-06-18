from activations import ReLU, LeakyReLU, Sigmoid, Tanh, Softmax

class Activation_switcher:

    def activation_switcher(activation):
        switcher = {
        'relu': ReLU(),
        'leaky_relu': LeakyReLU(),
        'sigmoid': Sigmoid(),
        'tanh': Tanh(),
        'softmax': Softmax()
        }
        return switcher.get(activation, None)

    def get_attribute(activation):
        switcher = {
            'relu': 'relu',
            'leaky_relu': 'leaky_relu',
            'sigmoid': 'sigmoid',
            'tanh': 'tanh',
            'softmax': 'softmax'
        }
        return switcher.get(activation, None)