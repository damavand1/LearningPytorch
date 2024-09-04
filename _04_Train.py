import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.nn import MSELoss
from torch.utils.data import random_split

from _01_Dataset_RealEstate import RealEstateDataset  # فرض کنیم کلاستان در فایل real_estate_dataset.py ذخیره شده است
from _02_NN_Regression import NeuralNetwork  # فرض کنیم کلاستان در فایل neural_network.py ذخیره شده است

# 1. Load Dataset
dataset_address = 'Datasets/RealEstate_Dataset.xlsx'  # آدرس فایل داده‌ها را تنظیم کنید
dataset = RealEstateDataset(dataset_address)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# 2. Create DataLoader
batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# 3. Initialize Model, Loss Function, Optimizer
input_features_count = 3  # تعداد ویژگی‌های ورودی (Area, Rooms, Age)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = NeuralNetwork(input_features_count).to(device)
#model = NeuralNetwork(input_features_count)
criterion = MSELoss()  # Mean Squared Error Loss for regression # میگن این ام اس ایی لاس برای رگرسیون مناسب هست
# اگر دسته بندی بود چیز دیگری استفاده میکردیم
optimizer = Adam(model.parameters(), lr=0.001)

# 4. Train the Model
def train_model(model, train_loader, criterion, optimizer, epochs=20):
    for epoch in range(epochs):
        # مد آموزشی مدل
        model.train()  # Set the model to training mode
        running_loss = 0.0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)  # Move to CUDA
            
            # Zero the parameter gradients
            # تنظیم مجدد گرادیان‌ها
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs) # پیش‌بینی خروجی مدل
            loss = criterion(outputs.squeeze(), targets)# محاسبه هزینه

            # Backward pass and optimize
            loss.backward()  # محاسبه گرادیان
            optimizer.step() # به‌روزرسانی پارامترهای مدل

            running_loss += loss.item()  # جمع‌کردن هزینه‌ها برای نمایش

        # Print epoch loss
        print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(train_loader)}")

# 5. Validate the Model
def validate_model(model, val_loader, criterion):
    model.eval()  # Set the model to evaluation mode
    val_loss = 0.0
    with torch.no_grad():  # No need to calculate gradients during validation
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)  # Move to CUDA
            outputs = model(inputs)
            loss = criterion(outputs.squeeze(), targets)
            val_loss += loss.item()

    print(f"Validation Loss: {val_loss/len(val_loader)}")

# Train and Validate the model
# اجرای آموزش
train_model(model, train_loader, criterion, optimizer, epochs=100)
validate_model(model, val_loader, criterion)

# 6. Save the Trained Model
# ذخیره مدل آموزش‌دیده
torch.save(model.state_dict(), 'real_estate_model.pth')
print("Model saved to real_estate_model.pth")
