import torch

environment = torch.tensor([[0, 1, 0],
                            [2, -10, 10]], torch.float)
Q_table = torch.zeros(6, 4)
gama = 0.1