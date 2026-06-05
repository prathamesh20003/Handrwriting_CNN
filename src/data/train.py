#import libraries 
import torch

from torch import optim
from torch import nn
from tqdm import tqdm
from load import train_loader
from models.cnnmodel import CNN 
       
       
device = "cuda" if torch.cuda.is_available() else "cpu"

model = CNN(in_channels=1, num_classes=10).to(device)



criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(model.parameters(), lr=0.001)


num_epochs=10
for epoch in range(num_epochs):
   print(f"Epoch [{epoch + 1}/{num_epochs}]")

   for batch_index, (data, targets) in enumerate(tqdm(train_loader)):
       data = data.to(device)
       targets = targets.to(device)
       scores = model(data)
       loss = criterion(scores, targets)
       optimizer.zero_grad()
       loss.backward()
       optimizer.step()

torch.save(model.state_dict(), 'MulticlassCNN.pth')
