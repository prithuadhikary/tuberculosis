import PIL.Image
import torch
import cv2
from torchvision.datasets import ImageFolder

from tuberculosis_detection import TuberculosisDetector, imageTransformer, image_dataset, test_dataset

india_tb_dataset = ImageFolder(
    root='data',
    transform=imageTransformer
)

test_dataset_loader = torch.utils.data.DataLoader(test_dataset, shuffle=True)

model = TuberculosisDetector()

model.load_state_dict(torch.load('trained_model.dict', weights_only=False))

accurate = 0
for step, (x, y) in enumerate(test_dataset_loader):
    with torch.no_grad():
        pred = model(x)
        pred = torch.round(pred).to(torch.int)
        if pred.item() == y:
            accurate += 1

print(f"Total accuracy: {accurate/len(test_dataset) * 100} %")

# model.eval()
#
# image = PIL.Image.open('data/TEST_nx18.jpg')
#
# image = imageTransformer(image)
#
# pred = model(image.reshape(1, 1, 100, 100))
#
# classes = image_dataset.find_classes('data')
#
# print(classes[0][torch.round(pred).to(torch.int)])