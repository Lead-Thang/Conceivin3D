import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import trimesh
import numpy as np
import os
from typing import Tuple, Optional

class ConceivoNet(nn.Module):
    def __init__(self, input_size=6, hidden_size=20, output_size=3):
        """
        Enhanced ConceivoNet for 3D editing.
        - input_size: Features (cost, material strength, time, vertex count, edge length, command embedding)
        - output_size: Transformation parameters (e.g., distance, cut position, slide distance)
        """
        super(ConceivoNet, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(hidden_size, hidden_size // 2)
        self.layer3 = nn.Linear(hidden_size // 2, output_size)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layer1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.layer2(x)
        x = self.relu(x)
        x = self.layer3(x)
        return x

class ArchitecturalDataset(Dataset):
    def __init__(self, csv_file: str, model_dir: str = "models", command_vocab: Optional[dict] = None):
        """
        Dataset with 3D mesh features and command embeddings.
        - csv_file: Path to architectural data CSV
        - model_dir: Directory containing STL files
        - command_vocab: Mapping of commands to embeddings (e.g., {"bevel-edges": [0.1, 0.2]})
        """
        data = pd.read_csv(csv_file)
        self.features = []
        self.labels = []
        self.command_vocab = command_vocab or {"bevel-edges": [0.1, 0.2, 0.3], "loop-cut-and-slide": [0.4, 0.5, 0.6]}

        for idx, row in data.iterrows():
            # Normalize architectural features
            cost = row['cost_million_usd'] / data['cost_million_usd'].max()
            strength = row['material_strength_mpa'] / data['material_strength_mpa'].max()
            time = row['time_months'] / data['time_months'].max()

            # Load and process 3D mesh (if available)
            model_path = os.path.join(model_dir, f"{row['id']}.stl") if 'id' in row else None
            mesh_features = [0.0, 0.0]  # Default if no mesh
            if model_path and os.path.exists(model_path):
                mesh = trimesh.load(model_path)
                # Ensure mesh is a Trimesh object before accessing geometry properties
                if isinstance(mesh, trimesh.Trimesh):
                    vertex_count = len(mesh.vertices) / 1000  # Normalize
                    # Use edges_unique if available, fallback to edges
                    edges = getattr(mesh, 'edges_unique', mesh.edges)
                    edge_length = np.mean([np.linalg.norm(e[1] - e[0]) for e in edges]) / 10  # Normalize
                    mesh_features = [vertex_count, edge_length]
                else:
                    vertex_count = 0.0  # Default if no vertices attribute

            # Command embedding (simplified)
            command = row.get('command', 'none')
            cmd_embedding = self.command_vocab.get(command, [0.0, 0.0, 0.0])

            # Combined features
            feature_vec = [cost, strength, time] + mesh_features + cmd_embedding
            self.features.append(feature_vec)

            # Labels: Transformation parameters (distance, cut_pos, slide_dist) or efficiency
            label = [row.get('bevel_distance', 0.0), row.get('cut_position', 0.0), row.get('slide_distance', 0.0)]
            self.labels.append(label)

        self.features = torch.tensor(self.features, dtype=torch.float32)
        self.labels = torch.tensor(self.labels, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.features[idx], self.labels[idx]

def train_model(model_dir: str = "models"):
    # Initialize model
    model = ConceivoNet()
    command_vocab = {"bevel-edges": [1.0, 0.0, 0.0], "loop-cut-and-slide": [0.0, 1.0, 0.0], "none": [0.0, 0.0, 0.0]}
    dataset = ArchitecturalDataset('architectural_data.csv', model_dir, command_vocab)
    dataloader = DataLoader(dataset, batch_size=5, shuffle=True)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    # Training loop
    num_epochs = 200
    for epoch in range(num_epochs):
        total_loss = 0
        for features, labels in dataloader:
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(dataloader)
        if (epoch + 1) % 20 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Average Loss: {avg_loss:.4f}')

    # Save model
    torch.save(model.state_dict(), "conceivo_model.pth")
    print("Model saved as conceivo_model.pth")
    return model

if __name__ == "__main__":
    # Train and test the model
    model = train_model()
    
    # Test inference on first sample
    dataset = ArchitecturalDataset('architectural_data.csv', 'models')
    sample_feature, _ = dataset[0]
    model.eval()
    with torch.no_grad():
        prediction = model(sample_feature.unsqueeze(0))
    print("Sample input features:", sample_feature)
    print("Model prediction:", prediction.squeeze().numpy())
