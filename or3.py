# -*- coding: utf-8 -*-

"""
PyTorch: Тензоры и автоград
В приведенных выше примерах нам пришлось вручную реализовать как прямой, так
и обратный проходы нашей нейронной сети. Реализация обратного прохода вручную
не представляет большого труда для небольшой двухуровневой сети, но может быстро
 стать очень сложной задачей для больших сложных сетей.

К счастью, мы можем использовать автоматическое дифференцирование для
автоматизации вычисления обратных проходов в нейронных сетях. Автоград пакет
в PyTorch обеспечивает именно эту функциональность. При использовании autograd
прямой проход вашей сети будет определять вычислительный граф ; узлы в графе
будут тензорами, а ребра будут функциями, которые производят выходные тензоры
из входных тензоров. Обратное распространение по этому графику позволяет легко
вычислять градиенты.

Звучит сложно, использовать на практике довольно просто. Каждый тензор
представляет собой узел в вычислительном графе. Если xэто тензор, который
 имеет, x.requires_grad=Trueто x.gradдругой тензор, содержащий градиент
 xотносительно некоторого скалярного значения.

Здесь мы используем PyTorch Tensors и autograd, чтобы реализовать нашу
подходящую синусоидальную волну на примере полинома третьего порядка; теперь
нам больше не нужно вручную реализовывать обратный проход по сети:
"""

import torch
import math

dtype = torch.float
device = torch.device("cpu")
# device = torch.device("cuda:0")  # Uncomment this to run on GPU

# Create Tensors to hold input and outputs.
# By default, requires_grad=False, which indicates that we do not need to
# compute gradients with respect to these Tensors during the backward pass.
x = torch.linspace(-math.pi, math.pi, 2000, device=device, dtype=dtype)
y = torch.sin(x)

# Create random Tensors for weights. For a third order polynomial, we need
# 4 weights: y = a + b x + c x^2 + d x^3
# Setting requires_grad=True indicates that we want to compute gradients with
# respect to these Tensors during the backward pass.
a = torch.randn((), device=device, dtype=dtype, requires_grad=True)
b = torch.randn((), device=device, dtype=dtype, requires_grad=True)
c = torch.randn((), device=device, dtype=dtype, requires_grad=True)
d = torch.randn((), device=device, dtype=dtype, requires_grad=True)

learning_rate = 1e-6
for t in range(2000):
    # Forward pass: compute predicted y using operations on Tensors.
    y_pred = a + b * x + c * x ** 2 + d * x ** 3

    # Compute and print loss using operations on Tensors.
    # Now loss is a Tensor of shape (1,)
    # loss.item() gets the scalar value held in the loss.
    loss = (y_pred - y).pow(2).sum()
    if t % 100 == 99:
        print(t, loss.item())

    # Use autograd to compute the backward pass. This call will compute the
    # gradient of loss with respect to all Tensors with requires_grad=True.
    # After this call a.grad, b.grad. c.grad and d.grad will be Tensors holding
    # the gradient of the loss with respect to a, b, c, d respectively.
    loss.backward()

    # Manually update weights using gradient descent. Wrap in torch.no_grad()
    # because weights have requires_grad=True, but we don't need to track this
    # in autograd.
    with torch.no_grad():
        a -= learning_rate * a.grad
        b -= learning_rate * b.grad
        c -= learning_rate * c.grad
        d -= learning_rate * d.grad

        # Manually zero the gradients after updating weights
        a.grad = None
        b.grad = None
        c.grad = None
        d.grad = None

print(f'Result: y = {a.item()} + {b.item()} x + {c.item()} x^2 + {d.item()} x^3')
