import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from model import SimpleCNN


def evaluate(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0
    class_correct = [0] * len(test_loader.dataset.classes)
    class_total = [0] * len(test_loader.dataset.classes)

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            for i in range(len(labels)):
                label = labels[i]
                class_correct[label] += predicted[i].eq(label).item()
                class_total[label] += 1

    accuracy = 100 * correct / total
    print(f"Test Accuracy: {accuracy:.2f}%")

    print("\nClass-wise Accuracy:")
    for i in range(len(test_loader.dataset.classes)):
        class_name = test_loader.dataset.classes[i]
        class_acc = 100 * class_correct[i] / class_total[i] if class_total[i] > 0 else 0
        print(f"Accuracy of {class_name}: {class_acc:.2f}%")


def evaluate_main(data_path):
    transform = transforms.Compose([transforms.Resize((224, 224)),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                         std=[0.229, 0.224, 0.225])])

    test_dataset = datasets.ImageFolder(root=data_path, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    model = SimpleCNN()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    model.load_state_dict(torch.load("model.pth"))

    evaluate(model, test_loader, device)


def run_evaluation_process(data_path):
    evaluate_main(data_path)
