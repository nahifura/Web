import image
from PIL import Image
from torch import argmax
import main
import os
import torch
from torchvision import transforms

root_dir = os.path.join('.','test')
classes_name = sorted(os.listdir(root_dir))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(classes_name)
weight_path = os.path.join('.','best_model.pth')
pre_dir = os.path.join('.','to_pre')
transform = transforms.Compose([transforms.Resize((224,224)),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                     std=[0.229, 0.224, 0.225])])


def match(idx):
    return classes_name[idx]


pre_net = main.Mymodel()
pre_net = pre_net.to(device)
pre_net.eval()
pre_net.load_state_dict(torch.load(weight_path, map_location=device))
print(list(pre_net.parameters()))


for img_path in os.listdir(pre_dir):
    try:
        join_path = os.path.join('.','to_pre',img_path)
        image = Image.open(join_path).convert('RGB')
    except:
        print(f"读取 {img_path} 失败")
        continue

    img_tensor = transform(image).unsqueeze(0).to(device)
    pre_output = torch.argmax(pre_net(img_tensor)).item()
    matched = match(pre_output)
    print(f'the name of {img_path} is {matched}')


