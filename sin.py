import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)
X = np.linspace(0, 1, 100).reshape(-1, 1)
y = np.sin(2 * np.pi * X)

def lossnum(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2) / 2
#激活函数
class Relu:
    def forward(self, x):
        self.x = x
        return np.maximum(0, x)

    def backward(self, grad):
        grad[self.x <= 0] = 0
        return grad

    def update(self, learn_rate):
        pass

class Mlp:
    def __init__(self, input, output):
        self.weights = np.random.normal(0, 0.1, (input, output))
        self.bias = np.random.normal(0, 0.1, (1, output))

    def forward(self, inputs):
        self.inputs = inputs
        linear_output = np.dot(inputs, self.weights) + self.bias
        return linear_output

    def backward(self, grad):
        self.grad_weights = np.dot(self.inputs.T, grad)
        self.grad_bias = np.sum(grad, axis=0, keepdims=True)
        return np.dot(grad, self.weights.T)

    def update(self, learn_rate):
        self.weights = self.weights - self.grad_weights * learn_rate
        self.bias = self.bias - self.grad_bias * learn_rate

def train(model, X, y, learn_rate=0.0015, epochs=2000):
    for epoch in range(epochs):
        inputs = X
        for module in model:
            inputs = module.forward(inputs)

        y_pred = inputs
        loss = lossnum(y, y_pred)
        grad = y_pred - y

        #从后往前
        for module in reversed(model):
            grad = module.backward(grad)

        for module in model:
            module.update(learn_rate)

        if epoch % 100 == 0:
            print(epoch,loss)

model1 = Mlp(1, 50)
model2 = Mlp(50, 100)
model3 = Mlp(100, 50)
model4 = Mlp(50, 1) #输出层

relu1 = Relu()
relu2 = Relu()
relu3 = Relu()

model = [model1, relu1, model2, relu2, model3, relu3, model4]

train(model, X, y, learn_rate=0.0015, epochs=2000)

inputs = X
for module in model:
    inputs = module.forward(inputs)
y_pred = inputs

plt.scatter(X, y, color='blue')
plt.plot(X, y_pred, color='red')
plt.show()
