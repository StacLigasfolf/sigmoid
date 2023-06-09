# sigmoid

## Пишем нейросеть на Python с нуля
Этот пример предназначен для полных новичков, не имеющих никакого опыта в машинном обучении.

## 1. Составные элементы: нейроны

![alt tag](https://media.proglib.io/posts/2020/09/24/4c33a6683394ddd46163a03cfe343216.png)
Внутри нейрона происходят три операции. Сначала значения входов умножаются на веса: 
 ```x1*w1 + x2*w2``` 
Затем взвешенные входы складываются, и к ним прибавляется значение порога b:
```x1*w1 + x2*w2 + b```
Наконец, полученная сумма проходит через функцию активации:
```y = f(x1*w1 + x2*w2 + b)```
Функция активации преобразует неограниченные значения входов в выход, имеющий ясную и предсказуемую форму. Одна из часто используемых функций активации – сигмоида:
![alt tag](https://media.proglib.io/posts/2020/09/24/82f3fddc563e76c7d4e469d6a53b9840.webp)
Сигмоида выдает результаты в интервале (0, 1). Можно представить, что она «упаковывает» интервал от минус бесконечности до плюс бесконечности в (0, 1): большие отрицательные числа превращаются в числа, близкие к 0, а большие положительные – к 1.

### Пишем код для нейрона

Настало время написать свой нейрон! Мы используем NumPy, популярную и мощную расчетную библиотеку для Python, которая поможет нам с вычислениями:

```
import numpy as np

def sigmoid(x):
  #Наша функция активации: f(x) = 1 / (1 + e^(-x))
  return 1 / (1 + np.exp(-x))

class Neuron:
  def __init__(self, weights, bias):
    self.weights = weights
    self.bias = bias

  def feedforward(self, inputs):
    # Умножаем входы на веса, прибавляем порог, затем используем функцию активации
    total = np.dot(self.weights, inputs) + self.bias
    return sigmoid(total)

weights = np.array([0, 1]) # w1 = 0, w2 = 1
bias = 4                   # b = 4
n = Neuron(weights, bias)

x = np.array([2, 3])       # x1 = 2, x2 = 3
print(n.feedforward(x))    # 0.9990889488055994 
```


## 2. Собираем нейронную сеть из нейронов

Нейронная сеть – это всего лишь несколько нейронов, соединенных вместе. Вот как может выглядеть простая нейронная сеть:
![alt tag](https://media.proglib.io/posts/2020/10/02/de81e6549b3e3c3bc1e3fdc78fe59f9c.png)
У этой сети два входа, скрытый слой с двумя нейронами (h1 и h2) и выходной слой с одним нейроном (o1). Обратите внимание, что входы для o1 – это выходы из h1 и h2. Именно это создает из нейронов сеть.

### Пишем код нейронной сети
```
import numpy as np
#Здесь код реализации неирона
class OurNeuralNetwork:
  def __init__(self):
    weights = np.array([0, 1])
    bias = 0

    # Используем класс Neuron из предыдущего раздела
    self.h1 = Neuron(weights, bias)
    self.h2 = Neuron(weights, bias)
    self.o1 = Neuron(weights, bias)

  def feedforward(self, x):
    out_h1 = self.h1.feedforward(x)
    out_h2 = self.h2.feedforward(x)

    # Входы для o1 - это выходы h1 и h2
    out_o1 = self.o1.feedforward(np.array([out_h1, out_h2]))

    return out_o1

network = OurNeuralNetwork()
x = np.array([2, 3])
print(network.feedforward(x)) # 0.7216325609518421
``` 
Мы снова получили 0.7216! Похоже, наша сеть работает.


## 3. Обучаем нейронную сеть
Давайте обучим нашу нейронную сеть предсказывать пол человека по его росту и весу.

Мы будем представлять мужской пол как 0, женский – как 1, а также сдвинем данные, чтобы их было проще использовать:
Имя	Вес (минус 135)	Рост (минус 66)	Пол
Алиса	-2	-1	1
Боб	25	6	0
Чарли	17	4	0
Диана	-15	-6	1

Потери
Прежде чем обучать нашу нейронную сеть, нам нужно как-то измерить, насколько "хорошо" она работает, чтобы она смогла работать "лучше". Это измерение и есть потери (loss).

Мы используем для расчета потерь среднюю квадратичную ошибку (mean squared error, MSE):

Все сделано по обучающему материалу Ильи Гинсбурга: 
https://proglib.io/p/pishem-neyroset-na-python-s-nulya-2020-10-07
