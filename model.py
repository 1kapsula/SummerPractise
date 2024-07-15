
import torch
import torchvision.models as models
from PIL import Image
from torchvision import transforms

# Загрузка предобученной модели ResNet18 с использованием новых аргументов 'weights'
model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

# Переключение модели в режим оценки (inference)
model.eval()

# Преобразование изображения в формат, подходящий для модели ResNet
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Загрузка изображения
img = Image.open("/Users/dudberoll/PycharmProjects/pythonProject6/data/depth_000_Color.png")
img_t = preprocess(img)
batch_t = torch.unsqueeze(img_t, 0)

# Предсказание класса
with torch.no_grad():
    output = model(batch_t)

# Вывод размерности выхода
print(output.shape)  # torch.Size([1, 1000])

# Преобразование выходов в вероятности (опционально)
probabilities = torch.nn.functional.softmax(output[0], dim=0)

# Загрузка меток классов ImageNet
with open("imagenet_classes.txt") as f:
    labels = [line.strip() for line in f.readlines()]


# Получение предсказанного класса
_, predicted_idx = torch.max(output, 1)
predicted_idx = predicted_idx.item()  # Получение значения индекса как целого числа
predicted_label = labels[predicted_idx]

print(f"Predicted label: {predicted_label}")
print(f"Predicted probability: {probabilities[predicted_idx].item()}")
