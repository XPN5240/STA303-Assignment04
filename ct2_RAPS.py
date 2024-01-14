import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from examples.common.dataset import build_dataset
from torchcp.classification.loss import ConfTr
from torchcp.classification.predictors import SplitPredictor
from torchcp.classification.scores import THR
from torchcp.classification.scores import RAPS
from torchcp.utils import fix_randomness

from torchvision import datasets, transforms 
import warnings
warnings.filterwarnings('ignore')

# 修改模型结构
class CIFARNet(nn.Module):
    def __init__(self):
        super(CIFARNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(64 * 8 * 8, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 8 * 8)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# 修改数据集加载和预处理
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
train_data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True, pin_memory=True)
test_data_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False, pin_memory=True)

def train(model, device, train_loader,criterion,  optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()


if __name__ == '__main__':
    alpha = 0.01
    num_trials = 5
    loss = "ConfTr"
    result = {}
    print(f"############################## {loss} #########################")
    
    predictor = SplitPredictor(score_function=RAPS(penalty=0.1))
    criterion = ConfTr(weight=0.01,
                        predictor=predictor,
                        alpha=0.05,
                        fraction=0.5,
                        loss_type="valid",
                        base_loss_fn=nn.CrossEntropyLoss())
        
    fix_randomness(seed=0)
    ##################################
    # Training a pytorch model
    ##################################
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
   
    cal_dataset, test_dataset = torch.utils.data.random_split(test_dataset, [5000, 5000])
    cal_data_loader = torch.utils.data.DataLoader(cal_dataset, batch_size=1600, shuffle=False, pin_memory=True)
    test_data_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1600, shuffle=False, pin_memory=True)
    
    model = CIFARNet().to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)
    for epoch in range(1, 10):
        train(model, device, train_data_loader, criterion, optimizer, epoch)
        
    
    score_function = RAPS(penalty=0.1)

    predictor = SplitPredictor(score_function, model)
    predictor.calibrate(cal_data_loader, alpha)                
    result = predictor.evaluate(test_data_loader)
    print(f"Result--Coverage_rate: {result['Coverage_rate']}, Average_size: {result['Average_size']}")