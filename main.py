import torch
#from torch import nn
from torch.utils.data import DataLoader
from _01_Dataset_RealEstate import RealEstateDataset

print(torch.__version__)


device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Using {device} device")

real_estate_dataset = RealEstateDataset('Datasets/RealEstate_Dataset.xlsx')


dataloader = DataLoader(real_estate_dataset, batch_size=4, shuffle=True)

print(real_estate_dataset)
print(dataloader)
