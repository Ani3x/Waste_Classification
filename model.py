import torch.nn as nn
import torch.optim as optim


class WasteNet(nn.Module):
    def __init__(self, input_size, num_classes):
        super(WasteNet, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Dropout(0.2), # Losowo wyłącza neurony, przez co teoretycznie wychodzi to na plus reszcie, ale nie wiem czy to takie skuteczne
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.Linear(64, 16),
            nn.ReLU(),
            nn.Linear(16, num_classes) # klasy: carboard, glass, metal, paper, plastic, trash
        )
    
    def forward(self, x):
        return self.fc(x)


def model_in(input_size=8, num_classes=6, learning_rate=0.001):
    # Inicjalizacja
    net = WasteNet(input_size=input_size, num_classes=num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=learning_rate)

    return net, criterion, optimizer
