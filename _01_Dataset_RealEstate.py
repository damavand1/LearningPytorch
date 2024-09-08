# https://pytorch.org/tutorials/beginner/basics/data_tutorial.html#creating-a-custom-dataset-for-your-files
# https://www.youtube.com/watch?v=AgPogrEtYWM&list=PLGTnpzmoSsuHgWueAN72dVSqesigKMBJh&index=7

# normalizing
# https://blog.faradars.org/%D9%86%D8%B1%D9%85%D8%A7%D9%84-%D8%B3%D8%A7%D8%B2%DB%8C-%D8%AF%D8%A7%D8%AF%D9%87-%DA%86%DB%8C%D8%B3%D8%AA/
 
import torch
from torch.utils.data import Dataset,random_split
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import pickle

class RealEstateDataset(Dataset):
    #def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
    def __init__(self, dataset_address, normalize=True):
        # print('aa')
        # Step 1: Load the Excel file into a Pandas DataFrame
        self.data=pd.read_excel(dataset_address,sheet_name="Regression",skiprows=0)

        # Print the column names to debug
        print("Columns in the dataset:", self.data.columns)

        if normalize:
            # نرمال‌سازی ویژگی‌ها و برچسب‌ها با استفاده از MinMaxScaler با بازه [-1, 1]
            # self.scaler = MinMaxScaler(feature_range=(-1, 1))

            # نرمال‌سازی ویژگی‌ها و برچسب‌ها با استفاده از MinMaxScaler با بازه پیش فرض [0, 1]
            #self.scaler =  MinMaxScaler()


            self.feature_scaler = MinMaxScaler()
            self.label_scaler = MinMaxScaler()


            # زمانی که می‌خواهید داده‌ها را نرمال‌سازی کنید، MinMaxScaler نیاز دارد که داده‌ها به شکل دو بعدی (یعنی یک ماتریس یا آرایه با دو بعد) باشند، حتی اگر فقط یک ویژگی داشته باشید.
            # area = self.data['Area'].values.reshape(-1, 1)
            # rooms = self.data['Rooms'].values.reshape(-1, 1)
            # age = self.data['Age'].values.reshape(-1, 1)
            # price = self.data['Price'].values.reshape(-1, 1)

            # # نرمال‌سازی
            # area_normalized = self.scaler.fit_transform(area)
            # rooms_normalized = self.scaler.fit_transform(rooms)
            # age_normalized = self.scaler.fit_transform(age)
            # price_normalized = self.scaler.fit_transform(price)

            # ترکیب همه ویژگی‌ها در یک آرایه
            features = self.data[['Area', 'Rooms', 'Age']].values
            labels = self.data['Price'].values.reshape(-1, 1)

            # نرمال‌سازی همه ویژگی‌ها با یک اسکیلر
            #features_normalized = self.scaler.fit_transform(features)
            #labels_normalized = self.scaler.fit_transform(labels)

            features_normalized = self.feature_scaler.fit_transform(features)
            labels_normalized = self.label_scaler.fit_transform(labels)

            # ذخیره اسکیلر با استفاده از pickle
            # with open('scaler.pkl', 'wb') as file:
            #     pickle.dump(self.scaler, file)

            # Save both feature and label scalers
            with open('TrainResults/feature_scaler.pkl', 'wb') as file:
                pickle.dump(self.feature_scaler, file)

            with open('TrainResults/label_scaler.pkl', 'wb') as file:
                pickle.dump(self.label_scaler, file)

            # #squeeze دوباره برشون میگردونیم به حالت یک بعدی
            # #وبعد تبدیل به tensor
            # area = torch.tensor(area_normalized.squeeze(), dtype=torch.float32)
            # rooms = torch.tensor(rooms_normalized.squeeze(), dtype=torch.float32)
            # age = torch.tensor(age_normalized.squeeze(), dtype=torch.float32)
            # price = torch.tensor(price_normalized.squeeze(), dtype=torch.float32)


            # Combine features into a single tensor
            # features = torch.stack([area, rooms, age], dim=1)
            # labels = price
            
            # تبدیل به تنسورهای PyTorch
            self.features = torch.tensor(features_normalized, dtype=torch.float32)
            self.labels = torch.tensor(labels_normalized.squeeze(), dtype=torch.float32)

            # self.features = features
            # self.labels = labels
        else:
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

 
