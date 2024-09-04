# https://pytorch.org/tutorials/beginner/basics/data_tutorial.html#creating-a-custom-dataset-for-your-files
# https://www.youtube.com/watch?v=AgPogrEtYWM&list=PLGTnpzmoSsuHgWueAN72dVSqesigKMBJh&index=7

import torch
from torch.utils.data import Dataset,random_split
import pandas as pd


class RealEstateDataset(Dataset):
    #def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
    def __init__(self, dataset_address):
        # print('aa')
        # Step 1: Load the Excel file into a Pandas DataFrame
        self.data=pd.read_excel(dataset_address,sheet_name="Regression",skiprows=0)

        # Print the column names to debug
        print("Columns in the dataset:", self.data.columns)

        # Step 2: Convert DataFrame columns into PyTorch tensors
        area = torch.tensor(self.data['Area'].values, dtype=torch.float32)
        rooms = torch.tensor(self.data['Rooms'].values, dtype=torch.float32)
        age = torch.tensor(self.data['Age'].values, dtype=torch.float32)
        price = torch.tensor(self.data['Price'].values, dtype=torch.float32)

        # Combine features into a single tensor
        features = torch.stack([area, rooms, age], dim=1)
        labels = price

        self.features = features
        self.labels = labels

        #print(self.data.shape)
        #print(features)
        #print(price)

        #self.DataSet_Record = self.data.iloc[:,:-1] # همه سطر ها و همه ستون ها بجز آخری
        #self.DataSet_Record_label= self.data.iloc[:, 1] # همه سطر ها و فقط آخرین ستون

       # print(self.data.shape)
       # ddd=pd.read_excel('Dataset.xlsx')
       # 

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return (self.features[index], self.labels[index])

 
