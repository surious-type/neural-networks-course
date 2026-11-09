from torch import nn


class MLP(nn.Module):
    def __init__(self, input_size=1, one=50, out=1, func=nn.ReLU):
        super().__init__()
        self.mlp1 = nn.Linear(input_size, one)
        self.mlp2 = nn.Linear(one, out)
        self.func = func()

    def forward(self, x):
        x = self.func(self.mlp1(x))
        return self.mlp2(x)


class MLP3(nn.Module):
    def __init__(self, input_size=1, one=40, two=10, out=1, func=nn.ReLU):
        super().__init__()
        self.mlp1 = nn.Linear(input_size, one)
        self.mlp2 = nn.Linear(one, two)
        self.mlp3 = nn.Linear(two, out)
        self.func = func()

    def forward(self, x):
        x = self.func(self.mlp1(x))
        x = self.func(self.mlp2(x))
        return self.mlp3(x)
