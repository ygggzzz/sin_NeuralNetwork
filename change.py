import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)
X = np.linspace(0, 1, 100).reshape(-1, 1)
y = np.sin(2 * np.pi * X)


def lossnum(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2) / 2

# 激活函数
class Relu:
    def forward(self, x):
        self.x = x
        return np.maximum(0, x)

    def backward(self, grad):
        grad_copy=np.array(grad,copy=True)
        grad_copy[self.x <= 0] = 0
        return grad_copy

    def update(self, learn_rate):
        pass


class Mlp:
    def __init__(self, input, output):
        self.input = input
        self.output = output
        self.weights = np.random.normal(0, 0.1, (input, output))
        self.bias = np.random.normal(0, 0.1, (1, output))

    def forward(self, inputs):
        self.inputs = inputs
        return np.dot(inputs, self.weights) + self.bias

    def backward(self, grad):
        grad_copy = np.array(grad, copy=True)
        input_copy = np.array(self.inputs, copy=True)

        input_copy = input_copy.reshape(-1, self.input, 1)
        grad_copy = grad_copy.reshape(-1, 1, self.output)

        self.grad_weights = np.matmul(input_copy, grad_copy)
        self.grad_weights = self.grad_weights.reshape(-1,self.input,self.output)
        self.grad_weights = self.grad_weights.mean(axis=0)
        #print(f'self.grad_weights ',self.grad_weights.shape)

        self.grad_bias = np.array(grad, copy=True)
        self.grad_bias = self.grad_bias.reshape(-1, self.output)
        self.grad_bias = self.grad_bias.mean(axis=0,keepdims=True)
        #print(f'self.grad_bias',self.grad_bias.shape)

        return np.dot(grad, self.weights.T)

    def update(self, learn_rate):
        self.weights = self.weights - learn_rate * self.grad_weights
        #print(f'self.weights',self.weights.shape )
        self.bias = self.bias - learn_rate * self.grad_bias
        #print(f'self.bias ',self.bias.shape )


def train(module_list, X, y, learn_rate=0.15, epochs=2000):
    for epoch in range(epochs):
        inputs = X
        for module in module_list:
            inputs = module.forward(inputs)

        y_pred = inputs
        loss = lossnum(y, y_pred)
        grad = y_pred - y

        # 从后往前
        for module in reversed(module_list):
            grad = module.backward(grad)


        for module in module_list:
            module.update(learn_rate)

        if epoch % 100 == 0:
            print(epoch, loss)


Mlp1 = Mlp(1, 50)
Mlp2 = Mlp(50, 100)
Mlp3 = Mlp(100, 50)
Mlp4 = Mlp(50, 1)

relu1 = Relu()
relu2 = Relu()
relu3 = Relu()

module_list = [Mlp1, relu1, Mlp2, relu2, Mlp3, relu3, Mlp4]

train(module_list, X, y, learn_rate=0.15, epochs=2000)

inputs = X
for module in module_list:
    inputs = module.forward(inputs)
y_pred = inputs

plt.scatter(X, y, color='blue')
plt.plot(X, y_pred, color='red')
plt.show()
