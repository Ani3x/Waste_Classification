import os
import json
import przerabianko
import model as codemodel
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


class CustomDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.classes = sorted(os.listdir(root_dir))
        self.samples = []

        # Robimy listę wszystkich plików raz na początku
        for target_class in self.classes:
            class_dir = os.path.join(root_dir, target_class)
            if not os.path.isdir(class_dir):
                continue
            for img_name in os.listdir(class_dir):
                self.samples.append((
                    os.path.join(class_dir, img_name),
                    self.classes.index(target_class)
                ))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        features = przerabianko.extract_features(img_path)
        
        # Obsługa błędnych zdjęć (np. jeśli OpenCV nie wczyta pliku)
        if features is None:
            return torch.zeros(self.number_features), torch.tensor(label)
            
        return features, torch.tensor(label)


if __name__ == "__main__":
    folder_path = 'Waste-Classification-1/train'

    dataset = CustomDataset(root_dir=folder_path)

    sample_features, _ = dataset[0]
    input_size = sample_features.shape[0] # Żeby był flex i nie trzeba by było liczyć tego co zmienioną cechę.

    dataloader = DataLoader(dataset, batch_size=4, shuffle=True) # Pakuje zdjęcia w paczki i daje w sieć na raz, co przyspiesza trenowanie
    
    model, criterion, optimizer = codemodel.model_in(input_size=input_size)
    
    for epoch in range(60):
        running_loss = 0.0
        for i, data in enumerate(dataloader):
            inputs, labels = data
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels.squeeze())
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch {epoch + 1}, Loss: {running_loss / len(dataloader)}")
    print('Finished Training')
    MODEL_PATH = "waste_model.pth"

    # Zapisujemy same wagi (state_dict)
    torch.save(model.state_dict(), MODEL_PATH)

    # Zapis klas, żeby było wiadamo jaki numer jakiej klasie jest przypisany
    with open("classes.json", "w") as f:
        json.dump(dataset.classes, f)

    print(f"Model saved as {MODEL_PATH}")
