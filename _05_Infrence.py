import torch
from torch import nn
import numpy as np

from _02_NN_Regression import NeuralNetwork  # فرض کنید کلاس مدل در این فایل است

# پارامترها و تنظیمات اولیه
input_features_count = 3  # تعداد ویژگی‌های ورودی (Area, Rooms, Age)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. بارگذاری مدل آموزش‌دیده
model = NeuralNetwork(input_features_count).to(device)
model.load_state_dict(torch.load('real_estate_model.pth'))
model.eval()  # تنظیم مدل به حالت ارزیابی

# 2. تابع پیش‌بینی (Inference)
def predict_price(area, rooms, age):
    # ایجاد ورودی به صورت یک numpy array
    input_data = np.array([[area, rooms, age]], dtype=np.float32)
    
    # نرمال‌سازی ورودی (در صورت نیاز، بسته به نحوه نرمال‌سازی در آموزش)
    # اینجا فرض می‌کنیم نرمال‌سازی بین 0 و 1 بوده است
    # توجه: اگر از MinMaxScaler استفاده کرده‌اید، باید از همان اسکیلر در اینجا استفاده کنید.
    # مثال فرضی برای نرمال‌سازی:
    # scaler = MinMaxScaler()
    # input_data = scaler.transform(input_data)
    
    # تبدیل ورودی به تنسور PyTorch و انتقال به دستگاه مناسب (CPU یا GPU)
    input_tensor = torch.tensor(input_data, device=device)

    # انجام پیش‌بینی
    with torch.no_grad():  # برای جلوگیری از محاسبه گرادیان‌ها
        output = model(input_tensor)
    
    # استخراج قیمت پیش‌بینی‌شده از تنسور خروجی
    predicted_price = output.cpu().item()  # انتقال به CPU و تبدیل به عدد
    
    return predicted_price

# 3. استفاده از تابع پیش‌بینی
# مثال استفاده:
area = 150.0  # مساحت
rooms = 3     # تعداد اتاق
age = 10      # سن ساختمان
predicted_price = predict_price(area, rooms, age)

print(f"Predicted Price: {predicted_price}")
