from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from rest_framework.parsers import JSONParser
import torch
from torchvision import transforms
from PIL import Image
import io
import numpy as np

# PyTorchのモデルのインポート
class CNN(torch.nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.fc1 = torch.nn.Linear(7*7*64, 128)
        self.fc2 = torch.nn.Linear(128, 10)  # 10クラス（0〜9）

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.max_pool2d(x, 2)
        x = torch.relu(self.conv2(x))
        x = torch.max_pool2d(x, 2)
        x = x.view(-1, 7*7*64)  # Flatten
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# モデルの読み込み
model = CNN()
model.load_state_dict(torch.load('mnist_cnn.pth'))
model.eval()

# 手書き文字の画像を受け取って推論するビュー
@csrf_exempt
def predict(request):
    if request.method == 'POST':
        # 画像を受け取る
        image_file = request.FILES['image']
        image = Image.open(image_file)

        # 画像の前処理
        transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),  # 1チャネル（グレースケール）
            transforms.Resize((28, 28)),  # MNISTのサイズにリサイズ
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))  # 正規化
        ])
        image = transform(image).unsqueeze(0)  # バッチ次元を追加

        # 推論
        with torch.no_grad():
            output = model(image)
            _, predicted = torch.max(output, 1)

        # 結果を返す
        return JsonResponse({'prediction': predicted.item()})
