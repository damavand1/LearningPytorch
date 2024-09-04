# https://pytorch.org/tutorials/beginner/basics/buildmodel_tutorial.html
# https://www.youtube.com/watch?v=YHhwv4fL2nA&list=PLGTnpzmoSsuHgWueAN72dVSqesigKMBJh&index=11
# Regression: پیش‌بینی مقادیر عددی پیوسته (مثل پیش‌بینی قیمت خانه).
# classification
import torch
from torch import nn

device = "cuda" if torch.cuda.is_available() else "cpu"

class NeuralNetwork(nn.Module):
    def __init__(self): # تعریف لایه های شبکه
        super().__init__()
        self.flatten = nn.Flatten() # داده‌های ورودی شبکه را از حالت دو بعدی (یا چند بعدی) به یک بردار یک‌بعدی تبدیل کند تا برای مراحل بعدی پردازش در شبکه‌ی عصبی آماده شود.
        self.linear_relu_stack = nn.Sequential(
            # نکته خیلی خیلی مهم در شبکه های عصبی این هست که لایه اول که 
            # input Layer
            #  نام دارد در اینجا 28*28 نورون دارد
            # و داده هایی که به این نورون ها داده میشود برای اینکه از نرون خارج شود تحت تابع فعال سازی
            # یا همان activation function قرار نمی گیرد
            # و دقیقاً همان مقداری که به نود ورودی دادیم در خروجی آن نود هم می آید
            # بر اساس جلسه هشتم هم روشین مصطفی آصفی NetSignals
            nn.Linear(28*28, 512), #این لایه وظیفه دارد که تمام ۷۸۴ نورون ورودی (که ممکن است پیکسل‌های یک تصویر باشند) را به ۵۱۲ نورون خروجی تبدیل کند و به هر نورون ورودی، یک وزن مشخص اختصاص دهد که این وزن‌ها در طول فرآیند آموزش شبکه یاد گرفته می‌شوند و به‌روزرسانی می‌شوند.---- سپس از فرمول y=xWT+by=xWT+b برای محاسبه خروجی استفاده می‌کند.
            nn.ReLU(), # Rectified Linear Unit (ReLU) activation function -> ReLU= max(0, x)
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )

    def forward(self, x): # ایکس را به عنوان ورودی میگیرد، که ظاهراً ورودی لایه ها هست
        # پس ورودی لایه را میگیرد، لایه را رویش اجرا می کند و خروجی می دهد
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits
    



print(f"Using {device} device")