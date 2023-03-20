from torch.autograd import Function

class ReverseGradient(Function):
    @staticmethod
    def forward(self, x):
        return x
    @staticmethod
    def backward(self, gradient):
        return (-gradient)
    