import torch
import torchvision.transforms as transforms

from pathlib import Path
from fastapi import FastAPI, UploadFile, File
from PIL import Image
from src.models.cnnmodel import CNN

app = FastAPI()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BASE_DIR = Path(__file__).resolve().parent.parent.parent
MODEL_PATH = BASE_DIR / "src/saved_model/MulticlassCNN.pth"
model = CNN(in_channels=1, num_classes=10)
model.load_state_dict(
    torch.load(MODEL_PATH, map_location=device)
)

model.to(device)
model.eval()

transform = transforms.Compose([
    transforms.Grayscale(1),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
])



@app.get("/")
def home():
    
    return {"message": f"MNIST CNN API running \n {model}"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image = Image.open(file.file)

    image = transform(image)

    t_image = torch.tensor(image)
    
    t_image = t_image.unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(t_image)

        probs = torch.softmax(output, dim=1)

        prediction = int(torch.argmax(probs, dim=1).item())

        confidence = float(probs[0, prediction].item())

    return {
        "digit": prediction,
        "confidence": confidence
    }
    