# Adapted from the WildPPG project:
# https://github.com/eth-siplab/WildPPG
# Original code licensed under GNU GPL-3.0.
# Modified for the Multi-site PPG project.
# See the repository LICENSE file for license details.

from torch.utils.data import Dataset

class base_loader(Dataset):
    def __init__(self, samples, labels, domains):
        self.samples = samples
        self.labels = labels
        self.domains = domains

    def __getitem__(self, index):
        sample, target, domain = self.samples[index], self.labels[index], self.domains[index]
        return sample, target, domain

    def __len__(self):
        return len(self.samples)


