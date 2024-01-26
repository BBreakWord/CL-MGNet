import numpy as np
import torch
from torch.utils.data.dataset import Dataset


class HSIDataset(Dataset):
    def __init__(self, dataset, transfor):
        super(HSIDataset, self).__init__()
        self.data = dataset[0].astype(np.float32)
        self.transformer = transfor
        self.labels = []
        for n in dataset[1]: self.labels += [int(n)]

    def __getitem__(self, index):
        img = torch.from_numpy(np.asarray(self.data[index,:,:,:]))
        label = self.labels[index]
        return img, label

    def __len__(self):
        return len(self.labels)

    def __labels__(self):
        return self.labels


if __name__ == "__main__":
    print(torch.cuda.is_available())