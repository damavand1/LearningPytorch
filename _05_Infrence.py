import torch
from torch import nn
import numpy as np
import pickle

from _02_NN_Regression import NeuralNetwork  # فرض کنید کلاس مدل در این فایل است

# پارامترها و تنظیمات اولیه
input_features_count = 3  # تعداد ویژگی‌های ورودی (Area, Rooms, Age)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. بارگذاری مدل آموزش‌دیده
model = NeuralNetwork(input_features_count).to(device)
model.load_state_dict(torch.load('TrainResults/real_estate_model.pth'))
model.eval()  # تنظیم مدل به حالت ارزیابی

# # 2. بارگذاری اسکیلر با pickle
# with open('scaler.pkl', 'rb') as file:
#     scaler = pickle.load(file)

# 2. بارگذاری اسکیلرهای ویژگی و برچسب با pickle
with open('TrainResults/feature_scaler.pkl', 'rb') as file:
    feature_scaler = pickle.load(file)

with open('TrainResults/label_scaler.pkl', 'rb') as file:
    label_scaler = pickle.load(file)


# 3. تابع پیش‌بینی (Inference)
def predict_price(area, rooms, age):
    # ایجاد ورودی به صورت یک numpy array
    input_data = np.array([[area, rooms, age]], dtype=np.float32)
    
    # نرمال‌سازی ورودی (در صورت نیاز، بسته به نحوه نرمال‌سازی در آموزش)
    # اینجا فرض می‌کنیم نرمال‌سازی بین 0 و 1 بوده است
    # توجه: اگر از MinMaxScaler استفاده کرده‌اید، باید از همان اسکیلر در اینجا استفاده کنید.
    # مثال فرضی برای نرمال‌سازی:
    # scaler = MinMaxScaler()
    # input_data = scaler.transform(input_data)


    # نرمال‌سازی ورودی‌ها با استفاده از اسکیلر بارگذاری‌شده
    input_data_normalized = feature_scaler.transform(input_data)
    
    # تبدیل ورودی به تنسور PyTorch و انتقال به دستگاه مناسب (CPU یا GPU)
    input_tensor = torch.tensor(input_data_normalized, device=device, dtype=torch.float32)

    # انجام پیش‌بینی
    with torch.no_grad():  # برای جلوگیری از محاسبه گرادیان‌ها
        output = model(input_tensor)
    
    # استخراج قیمت پیش‌بینی‌شده از تنسور خروجی
    #predicted_price = output.cpu().item()  # انتقال به CPU و تبدیل به عدد
    
    # تبدیل خروجی نرمال شده به قیمت واقعی با استفاده از label_scaler
    predicted_price_normalized = output.cpu().numpy().reshape(-1, 1)
    predicted_price = label_scaler.inverse_transform(predicted_price_normalized)
    
    return predicted_price.item()

    # return predicted_price

# 3. استفاده از تابع پیش‌بینی
# مثال استفاده:
area = 206.0  # مساحت
rooms = 3     # تعداد اتاق
age = 2    # سن ساختمان
predicted_price = predict_price(area, rooms, age)

print(f"Predicted Price: {predicted_price}")
