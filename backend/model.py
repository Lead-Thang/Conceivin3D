# D:\Conceivin3D\backend/model.py
import torch
import torch.nn as nn

class ConceivoNet(nn.Module):
    def __init__(self, input_size=3, hidden_size=10, output_size=1):
        super(ConceivoNet, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.layer1(x)
        x = self.relu(x)
        x = self.layer2(x)
        return x

if __name__ == "__main__":
    model = ConceivoNet()
    torch.save(model.state_dict(), "conceivo_model.pth")
    print("Model saved as conceivo_model.pth")