import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import Dataset, DataLoader

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

class ArchitecturalDataset(Dataset):
    def __init__(self, csv_file):
        data = pd.read_csv(csv_file)
        self.features = torch.tensor(data[['cost_million_usd', 'material_strength_mpa', 'time_months']].values, dtype=torch.float32)
        self.labels = torch.tensor(data['efficiency'].values, dtype=torch.float32).reshape(-1, 1)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

def train_model():
    model = ConceivoNet()
    dataset = ArchitecturalDataset('architectural_data.csv')
    dataloader = DataLoader(dataset, batch_size=5, shuffle=True)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    num_epochs = 100
    for epoch in range(num_epochs):
        for features, labels in dataloader:
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        if (epoch + 1) % 20 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

    torch.save(model.state_dict(), "conceivo_model.pth")
    print("Model saved as conceivo_model.pth")

if __name__ == "__main__":
    train_model()