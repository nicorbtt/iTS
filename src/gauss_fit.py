import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

class GaussFitter(nn.Module):
    def __init__(self, B=200):
        super(GaussFitter, self).__init__()
        self.mean = nn.Linear(B, 1)
        self.stddev = nn.Linear(B, 1)
        self.softplus = nn.Softplus()

    def forward(self, x):
        mean = self.mean(x)
        sd = self.softplus(self.stddev(x))
        return mean, sd



if __name__ == "__main__":

    B = 200
    model = GaussFitter(B)

    true_mean, true_sd = 7, 2
    print(f"\nmean_true={true_mean} \t sd_true={true_sd}\n")

    samples = torch.normal(true_mean, true_sd, size=(1_000, B))

    dataset = TensorDataset(samples)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    nll_loss_fn = nn.GaussianNLLLoss()
    optimizer = optim.Adam(model.parameters())

    means = []
    stddevs = []
    for epoch in range(501):
        for batch in dataloader:
            optimizer.zero_grad()
            inputs = batch[0]
            mean, stddev = model(inputs)
            means.append(torch.mean(mean).item())
            stddevs.append(torch.mean(stddev).item())
            loss = nll_loss_fn(inputs, mean, stddev)
            loss.backward()
            optimizer.step()
        if epoch % 100 == 0:
            print(f'Epoch {epoch}, Loss: {loss.item()}')
        epoch = epoch + 1

print(f"\nmean_hat={means[-1]:.2f} \t sd_hat={np.sqrt(stddevs[-1]):.2f}\n")