import torch
import os
import przerabianko
import model
from main import CustomDataset
from torch.utils.data import DataLoader


classes = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]
test_dir = 'Waste-Classification-1/test'
model_path = 'waste_model.pth'


def evaluate():

    # 1. Przygotowanie danych testowych
    test_dataset = CustomDataset(root_dir=test_dir)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    correct = 0
    total = 0
    
    class_correct = {cls: 0 for cls in classes}
    class_total = {cls: 0 for cls in classes}

    # 2. Konfiguracja
    sample_features, _ = test_dataset[0]
    input_size = sample_features.shape[0] # Żeby był flex i nie trzeba by było liczyć tego co zmienioną cechę.
    
    # Nazwy klas - muszą być w tej samej kolejności co przy trenowaniu
    num_classes = len(classes)

    # 3. Inicjalizacja i wczytanie modelu
    net, _, _ = model.model_in(input_size=input_size, num_classes=num_classes)
    
    if not os.path.exists(model_path):
        print(f"Błąd: Nie znaleziono pliku modelu {model_path}!")
        return

    net.load_state_dict(torch.load(model_path))
    net.eval() # BARDZO WAŻNE: przełącza model w tryb testowania
    print("Model wczytany pomyślnie.")

    print(f"Rozpoczynam testowanie na {len(test_dataset)} obrazach...")

    # 4. Pętla testująca
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = net(inputs)
            
            # Wybieramy klasę z największym prawdopodobieństwem (argmax)
            _, predicted = torch.max(outputs.data, 1)
            
            total += 1
            label_idx = labels.item()
            pred_idx = predicted.item()

            if pred_idx == label_idx:
                correct += 1
                class_correct[classes[label_idx]] += 1
            
            class_total[classes[label_idx]] += 1

    # 5. Wyświetlenie wyników
    accuracy = 100 * correct / total
    print("\n" + "="*30)
    print(f"WYNIK OGÓLNY: {correct}/{total} ({accuracy:.2f}%)")
    print("="*30)
    
    print("\nSkuteczność dla poszczególnych klas:")
    for cls in classes:
        if class_total[cls] > 0:
            cls_acc = 100 * class_correct[cls] / class_total[cls]
            print(f"- {cls:10}: {class_correct[cls]}/{class_total[cls]} ({cls_acc:.2f}%)")
        else:
            print(f"- {cls:10}: Brak obrazów testowych")

if __name__ == "__main__":
    evaluate()
