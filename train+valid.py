import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import f1_score
from sklearn.model_selection import KFold
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

transform = transforms.Compose([
   transforms.Resize((224, 224)),   # B0 chuẩn 224x224
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

dataset = torchvision.datasets.ImageFolder(root='train', transform=transform)
# trainloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

# {'bread': 0, 'dairy': 1, 'dessert': 2, 'egg': 3, 'fried': 4, 'meat': 5, 'noodles-pasta': 6, 'rice': 7, 'seafood': 8, 'soup': 9, 'vegetable-fruit': 10}





class Net(nn.Module):
    def __init__(self, input_size=(3, 224, 224)):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(4, 4)
        self.conv2 = nn.Conv2d(6, 16, 5)

        # Tính kích thước flatten động bằng một dummy forward pass (CPU)
        with torch.no_grad():
            dummy = torch.zeros(1, *input_size)
            x = self.pool(F.relu(self.conv1(dummy)))
            x = self.pool(F.relu(self.conv2(x)))
            flatten_dim = x.numel()  # channels * H * W

        self.fc1 = nn.Linear(flatten_dim, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 11)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)  # flatten tất cả ngoại trừ batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def train(dataloader, model, criterion, optimizer,device):
    running_loss = 0.0
    model.train()
    for i, data in enumerate(dataloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data[0].to(device), data[1].to(device)                     
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        # print statistics
        running_loss += loss.item()
        if i % 100 == 0:    # print every 2 mini-batches
            print(f'{i + 1:5d}] loss: {running_loss / 2000:.3f}')
            running_loss = 0.0

    print('Finished Training')

def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
            all_preds.extend(pred.argmax(1).cpu().numpy())
            all_labels.extend(y.cpu().numpy())
    test_loss /= num_batches
    correct /= size
    f1_macro = f1_score(all_labels, all_preds, average='macro')
    f1_micro = f1_score(all_labels, all_preds, average='micro')
    f1= f1_score(all_labels, all_preds, average=None)
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    print(f"F1 macro: {f1_macro:.4f}, F1 micro: {f1_micro:.4f}")
    print(f1)
    

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    # weights=EfficientNet_B0_Weights.IMAGENET1K_V1
    # model = efficientnet_b0(weights)  #Mettez weight dans le parametre pour utiliser le pretrained
    # num_classes = 11
    # model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    model = Net()

    # model.load_state_dict(torch.load("modelCNN.pth", weights_only=True))
    model=model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    
    
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)

    for fold, (train_idx, val_idx) in enumerate(kfold.split(dataset)):
        print(f'FOLD {fold}')
        print('--------------------------------')
        # Create subset cho train và val
        train_subset = torch.utils.data.Subset(dataset, train_idx)
        val_subset   = torch.utils.data.Subset(dataset, val_idx)

        # Create dataloader
        train_loader = torch.utils.data.DataLoader(train_subset, batch_size=32, shuffle=True)
        val_loader   = torch.utils.data.DataLoader(val_subset, batch_size=32, shuffle=False)
        for epoch in range(3):  # loop over the dataset multiple times
            print(f"Epoch {epoch+1}\n-------------------------------")
            train(train_loader, model, criterion, optimizer,device)
            test(val_loader, model, criterion)
    print("Done!")
    torch.save(model.state_dict(), "modelCNN.pth")
    print("Saved PyTorch Model State to modelCNNnew.pth")