import pandas as pd
import numpy as np
data = pd.read_csv(r'/Users/sharini/Downloads/Bank_Personal_Load_Modelling.csv')
data.head()

data = data.drop(['ID', 'Personal Loan'], axis = 1)
data.head()
data = data.dropna()

x = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state = 0)

class NN(nn.Module):
    def__init__(self):
        super().__init__()
        self.hidden1 = nn.Linear(8, 12)
        self.act1 = nn.ReLu()
        self.hidden2 = nn.Linear(12, 8)
        self.act2 = nn.ReLu()
        self.output = nn.Linear(8, 1)
        self.activate_output = nn.Sigmoid()

    def forward(self, i):
        i = self.act1(self.hidden1(i))
        i = self.act2(self.hidden2(i))
        i = self.activate_output(self.output(i))
        return i


import random

n = 20
l1 = 1
l2 = 2
ru = 0.5
q = 1
t1 = 0.5
t2 = 0.5

epochs = 100
success = [0 for x in range(epochs)]

class ACO():
    def __init__(self, n, epochs, q, l1, l2, t1, t2, alpha, beta):
        self.n = n
        self.epochs = epochs
        self.q = q
        self.t1 = t1
        self.t2 = t2
        self.alpha = alpha
        self.beta = beta
        self.accumulation = None
        self.global_best = None

def compute_probability(t1, t2):
    return t1/(t1 + t1), t2/(t1 + t1)

def sel_path(p1, p2):
    if p1 > p2:
        return 1
    if p1 < p2:
        return 2
    if p1 == p2:
        return random.choice([1, 2])

def upd_accumulation(id):
    global t1
    global t2
    if id == 1:
        t1 += q/l1
        return t1
    if id == 2:
        t2 += q/l2
        return t2

def upd_evaporation():
    global t1
    global t2
    t1 *= (1-ru)
    t2 *= (1-ru)
    return t1, t2

for epoch in range(epochs - 1):
    temp = 0
    for ant in range(N - 1):
        p1, p2 = compute_probability(t1, t2)
        sel_path = sel_path(p1, p2)
        if sel_path == 1:
            temp += 1
        upd_accumulation(sel_path)
        upd_evaporation()

    success[epoch] = temp

model = NN()
optimizer = ACO(n = 20, epochs = 100, q = 1, l1 = 1, l2 = 2, t1 = 0.5, t2 = 0.5, alpha = 1, beta = 2)
loss = torch.nn.MSELoss()
print("epoch: {} ; loss = {loss}")
