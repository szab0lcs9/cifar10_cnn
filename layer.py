class Layer:
    def __init__(self):
        self.input = None
        self.output = None

    def forward(self, input):
        pass
    
    def backward(self, output_error, learning_rate):
        pass
