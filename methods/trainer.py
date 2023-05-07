import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
from tabulate import tabulate
import numpy as np

# Set random seeds for reproducibility
torch.manual_seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

if torch.cuda.is_available():
    device = torch.device("cuda")
    print("Running on the GPU")
else:
    device = torch.device("cpu")
    print("Running on the CPU")

def train(model, train_loader, test_loader, optimizer, criterion, n_epochs, scheduler, early_stopping_patience):
    # Initialize lists to store the metrics
    train_losses = []
    train_accs = []
    test_losses = []
    test_accs = []
    black_holes_accs = []
    sphalerons_accs = []
    test_precisions = []
    test_recalls = []

    best_val_loss = float('inf')
    best_model_state_dict = None
    no_improvement_count = 0

    metrics_table = []

    for epoch in range(n_epochs):
        # Train
        model.train()  # set the model to train mode
        train_loss = 0
        correct = 0
        total = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device).float(), labels.to(device).long()  # Mode inputs and labels to the device
            optimizer.zero_grad()  # Zero the gradients
            outputs = model(inputs).float()  # Pass inputs to the model and get the outputs
            loss = criterion(outputs, labels)  # calculate the loss
            loss.backward()  # Backpropogate the loss
            optimizer.step()  # Update the model parameters

            train_loss += loss.item()  # Accumulate the trainings loss
            _, predicted = outputs.max(1)  # Get the predictions
            total += labels.size(0)  # Accumulate the number of examples
            correct += predicted.eq(labels).sum().item()  # Accumulate the number of correct predictions

        train_acc = 100 * correct / total  # Calculate the training accuracy
        train_loss /= len(train_loader)  # Average the training loss

        # Test
        model.eval()  # Set the model to evaluation mode
        test_loss = 0
        correct = 0
        total = 0
        black_holes_correct = 0
        sphalerons_correct = 0
        black_holes_total = 0
        sphalerons_total = 0
        all_preds = []
        all_labels = []
        val_loss = 0

        with torch.no_grad():  # Disable gradient computation
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device).float(), labels.to(device).long()  # Move inputs and labels to device
                outputs = model(inputs).float()  # Pass inputs to the model and get the outputs

                loss = criterion(outputs, labels)  # Calculate the loss

                val_loss += loss.item()
                test_loss += loss.item()
                _, predicted = outputs.max(1)  # Get the predictions
                total += labels.size(0) # Accumulate the number of examples
                correct += predicted.eq(labels).sum().item() # Accumulate the number of correct predictions
                # Separate accuracies for black holes and sphalerons
            black_holes_correct += (predicted * labels).sum().item()
            sphalerons_correct += ((1 - predicted) * (1 - labels)).sum().item()
            black_holes_total += labels.sum().item()
            sphalerons_total += (1 - labels).sum().item()

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

        test_acc = 100 * correct / total
        black_holes_acc = 100 * black_holes_correct / black_holes_total
        sphalerons_acc = 100 * sphalerons_correct / sphalerons_total
        test_loss /= len(test_loader)
        # Calculate precision and recall
        precision, recall, _, _ = precision_recall_fscore_support(all_labels, all_preds, average=None)

        # Print results
        print(f"Epoch: {epoch}/{n_epochs}")
        print(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.2f}%")
        print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.2f}%")

        # Append the metrics to the lists
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        test_losses.append(test_loss)
        test_accs.append(test_acc)
        black_holes_accs.append(black_holes_acc)
        sphalerons_accs.append(sphalerons_acc)
        test_precisions.append(precision)
        test_recalls.append(recall)

        metrics_table.append([
            epoch, train_loss, train_acc, test_loss, test_acc,
            black_holes_acc, sphalerons_acc, *precision, *recall
        ])

        scheduler.step(val_loss)  # Update learning rate using the scheduler
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state_dict = model.state_dict()
            no_improvement_count = 0
        else:
            no_improvement_count += 1

        if no_improvement_count >= early_stopping_patience:
            print(f"Early stopping at epoch {epoch}")
            break

    # Load the best model state
    model.load_state_dict(best_model_state_dict)

    # Plot the metrics
    plt.figure(figsize=(18, 18))
    plt.subplot(3, 3, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(test_losses, label='Test Loss')
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Train and Test Loss')

    plt.subplot(3, 3, 2)
    plt.plot(train_accs, label='Train Accuracy')
    plt.plot(test_accs, label='Test Accuracy')
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Train and Test Accuracy')

    plt.subplot(3, 3, 3)
    plt.plot(black_holes_accs, label='Black Holes Accuracy')
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Black Holes Accuracy')

    plt.subplot(3, 3, 4)
    plt.plot(sphalerons_accs, label='Sphalerons Accuracy')
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Sphalerons Accuracy')
    
    plt.subplot(3, 3, 5)
    plt.plot(test_precisions, label='Test Precision')
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Precision')
    plt.title('Test Precision')

    plt.subplot(3, 3, 6)
    plt.plot(test_recalls, label='Test Recall')
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Recall')
    plt.title('Test Recall')

    plt.tight_layout()
    plt.show()
    
    # Print tabular data
    headers = [
        "Epoch", "Train Loss", "Train Acc", "Test Loss", "Test Acc",
        "BH Acc", "Sph Acc", "BH Precision", "Sph Precision", "BH Recall", "Sph Recall"
    ]

    print(tabulate(metrics_table, headers=headers, floatfmt=".2f"))

    metrics_table_np = np.array(metrics_table)
    mean = np.mean(metrics_table_np[:, 1:], axis=0)
    std_dev = np.std(metrics_table_np[:, 1:], axis=0)
    mean_std_row = np.hstack((['Mean ± Std Dev'], ['{:.2f} ± {:.2f}'.format(m, s) for m, s in zip(mean, std_dev)]))
    mean_std_row = np.hstack((mean_std_row[0:1], ['{:.2f} ± {:.2f}'.format(mean[0], std_dev[0])], [''], mean_std_row[2:]))
    print(tabulate([mean_std_row], headers=headers))
    