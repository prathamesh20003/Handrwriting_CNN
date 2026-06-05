import torch

from train import model
from load import test_loader 
from torchmetrics.classification import (Accuracy, Recall, Precision)

#Evaluation metrics
acc = Accuracy(task="multiclass",num_classes=10)
precision = Precision(task="multiclass", num_classes=10)
recall = Recall(task="multiclass", num_classes=10)

model.eval()
with torch.no_grad():
   for images, labels in test_loader:
       outputs = model(images)
       _, preds = torch.max(outputs, 1)
       acc(preds, labels)
       precision(preds, labels)
       recall(preds, labels)

test_accuracy = acc.compute()
print(f"Test accuracy: {test_accuracy}")

