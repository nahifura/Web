from fastapi import FastAPI, Form, HTTPException, UploadFile,File
from calc_logic import perform_calculation
import uvicorn
import torch
import torchvision
from torchvision import transforms
import os
import io
from PIL import Image

from model import res_pre

app = FastAPI()

#======================================================================================

# 算数的计算器接口

@app.post("/calculate")
def calculate(
    Operation: str = Form(...),
    Data_A: float = Form(...),
    Data_B: float = Form(...)
):
    try:
        result = perform_calculation(Operation, Data_A, Data_B)
        return {"Result": result}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"服务器内部错误: {str(e)}")

#======================================================================================
#初始化大模块


print("正在初始化模型和配置...")
from model.resnet import MyResNet18
import model.resnet_copy as resnet_copy
import model.res_pre

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
classes_name = sorted(['daisy', 'dandelion', 'rose', 'sunflowers', 'tulip'])
print(classes_name)
weight_path = os.path.join('model', 'res_best_model.pth')

transform = transforms.Compose([transforms.Resize((224, 224)),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                         std=[0.229, 0.224, 0.225])])

pre_net = resnet_copy.MyResNet18()
pre_net = pre_net.to(device)
pre_net.eval()

try:
    pre_net.load_state_dict(torch.load(weight_path, map_location=device))
    print("模型加载成功！")
except Exception as e:
    print(f"模型加载失败，请检查权重路径: {e}")
    print(f"加载的路劲是: {os.path.abspath(weight_path)}")



#=============================================================================================
# 预测大模块
# 接受信息
@app.post("/predict")
async def predict_on(
        file: UploadFile = File(...),
        operation: str = Form("default")
):
    # 1. 检查文件格式
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="格式错误：请上传JPG格式图片")

    try:
        # 2. 把任务打包给res_pre里的predict函数实现
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert('RGB')

        prediction = res_pre.predict(pre_net, image)


        # 3. 响应：返回 JSON
        return {
            "code": 200,
            "msg": "预测成功",
            "filename": file.filename,
            "operation_used": operation,
            "predicted_class": prediction[0]
        }

    except Exception as e:
        # 捕获预处理和模型推理异常
        raise HTTPException(status_code=500, detail=f"模型预测失败: {str(e)}")


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)