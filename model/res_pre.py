from PIL import Image
from torch import argmax
import model.main
import os
import torch
from torchvision import transforms

from model import resnet_copy

transform = transforms.Compose([transforms.Resize((224, 224)),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                    std=[0.229, 0.224, 0.225])])
classes_name = sorted(['daisy', 'dandelion', 'rose', 'sunflowers', 'tulip'])

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



def match(idx):
    return classes_name[idx]



def predict(pre_net,img):

    pre_net.eval()
    img_tensor = transform(img).unsqueeze(0).to(device)
    pre_output = torch.argmax(pre_net(img_tensor)).item()
    matched = match(pre_output)

    return matched
