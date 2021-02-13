# -*- coding: utf-8 -*-
import numpy as np
import math

"""
PyTorch: Тензоры
Numpy отличный фреймворк, но он не может использовать графические процессоры
для ускорения своих численных вычислений. Для современных глубоких нейронных
сетей графические процессоры часто обеспечивают ускорение в 50 раз или больше ,
поэтому, к сожалению, numpy недостаточно для современного глубокого обучения.

Здесь мы представляем самуfю фундаментальную концепцию PyTorch: тензор . PyTorch
Tensor концептуально идентичен массиву numpy: Tensor - это n-мерный массив, а
PyTorch предоставляет множество функций для работы с этими тензорами. За
кулисами тензорные элементы могут отслеживать вычислительный график и градиенты,
но они также полезны как универсальный инструмент для научных вычислений.

Также в отличие от numpy, PyTorch Tensors может использовать графические
процессоры для ускорения своих числовых вычислений. Чтобы запустить PyTorch
Tensor на GPU, вам просто нужно указать правильное устройство.

Здесь мы используем тензоры PyTorch, чтобы подогнать полином третьего порядка
к синусоидальной функции. Как и в приведенном выше примере numpy, нам нужно
вручную реализовать прямой и обратный проходы по сети:
"""

# Create random input and output data
# Ստեղծեք պատահական մուտքային և ելքային տվյալներ
x = np.linspace(-math.pi, math.pi, 2000)
y = np.sin(x)

# Randomly initialize weights
# Պատահականորեն նախանշեք կշիռները
a = np.random.randn()
b = np.random.randn()
c = np.random.randn()
d = np.random.randn()

learning_rate = 1e-6
for t in range(2000):
    # Փոխանցել փոխանցում. Հաշվարկել կանխատեսված y- ն
    # Forward pass: compute predicted y
    # y = a + b x + c x^2 + d x^3
    y_pred = a + b * x + c * x ** 2 + d * x ** 3

    # Հաշվարկել և տպել կորուստը
    # Compute and print loss
    loss = np.square(y_pred - y).sum()
    if t % 100 == 99:
        print(t, loss)

    # Backprop to compute gradients of a, b, c, d with respect to loss
    # Պատկեր ՝ a, b, c, d գրադիենտների կորստի նկատմամբ հաշվարկելու համար
    grad_y_pred = 2.0 * (y_pred - y)
    grad_a = grad_y_pred.sum()
    grad_b = (grad_y_pred * x).sum()
    grad_c = (grad_y_pred * x ** 2).sum()
    grad_d = (grad_y_pred * x ** 3).sum()

    # Update weights
    # Թարմացնել կշիռները
    a -= learning_rate * grad_a
    b -= learning_rate * grad_b
    c -= learning_rate * grad_c
    d -= learning_rate * grad_d

print(f'Result: y = {a} + {b} x + {c} x^2 + {d} x^3')
