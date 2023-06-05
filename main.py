import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


class Neuron:
    def __init__(self, weights, bais):
        self.weights = weights
        self.bais = bais

    def feedforward(self, inputs):
        # Умножаем входы на веса, прибавляем порог, затем используем функцию активации
        total = np.dot(self.weights, inputs) + self.bais
        return sigmoid(total)


weights = np.array([0, 1])  # w1 = 0, w2 = 1
bais = 4  # b = 4
n = Neuron(weights, bais)

x = np.array([2, 3])  # x1 = 2, x2 = 3
print(f'n = {n.feedforward(x)}')


class OurNeuralNetwork:
    '''
    Нейронная сеть с:
      - 2 входами
      - скрытым слоем с 2 нейронами (h1, h2)
      - выходным слоем с 1 нейроном (o1)
    Все нейроны имеют одинаковые веса и пороги:
      - w = [0, 1]
      - b = 0
    '''

    def __init__(self):
        weights = np.array([0, 1])
        bais = 0

        self.h1 = Neuron(weights, bais)
        self.h2 = Neuron(weights, bais)
        self.o1 = Neuron(weights, bais)

    def feedforward(self, x):
        out_h1 = self.h1.feedforward(x)
        out_h2 = self.h2.feedforward(x)

        out_o1 = self.o1.feedforward(np.array([out_h1, out_h2]))
        return out_o1


network = OurNeuralNetwork()
x = np.array([2, 3])
print(f'o1 = {network.feedforward(x)}')
