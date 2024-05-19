import torch
import torchvision
import torchvision.transforms as transforms
from PIL import Image
import numpy as np

from PPIDSG.models import (
    Generator,
    Discriminator,
    AutoEncoder_VGG,
    VGG16_classifier,
    AutoEncoder_VGG_mnist,
    VGG16_classifier_mnist,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
G = Generator(3, 32, 3, 6)

G.load_state_dict(
        torch.load("./cifar_model/" + "generator_param.pkl")
    )

G.to(device)

transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

cifar_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)

data_loader = torch.utils.data.DataLoader(cifar_dataset, batch_size=1, shuffle=True)

for images, labels in data_loader:
    image = images[0]
    image = image.detach().cpu().numpy().astype(np.uint8)
    image = Image.fromarray(np.transpose(image, (1, 2, 0)))
    image.save("original_image.png")
    break

image_tensor = image.unsqueeze(0).to(device)

_,generated_image = G(image_tensor)

generated_image = (generated_image + 1) * 127.5
generated_image = generated_image.cpu().squeeze().detach().numpy().astype(np.uint8)
generated_image = Image.fromarray(np.transpose(generated_image, (1, 2, 0)))

generated_image.save("generated_image.png")
