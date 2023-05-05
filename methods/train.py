import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import precision_recall_fscore_support

if torch.cuda.is_available():
    device = torch.device("cuda")
    print("Running on the GPU")
else:
    device = torch.device("cpu")
    print("Running on the CPU")

def train(model, train_loader, test_loader, optimizer, criterion, n_epochs):
    
    # Initialize lists to store the metrics
    train_losses = []
    train_accs = []
    test_losses = []
    test_accs = []
    black_holes_accs = []
    sphalerons_accs = []
    test_precisions = []
    test_recalls = []
    
    metrics_per_epoch = []
    
    for epoch in range(n_epochs):
        # Train
        model.train() # set the model to train mode
        train_loss = 0
        correct = 0
        total = 0
    
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device).float(), labels.to(device).long() # Move inputs and labels to the device
            optimizer.zero_grad() # Zero the gradients
            outputs = model(inputs).float() # Pass inputs to the model and get the outputs
            loss = criterion(outputs, labels) # Calculate the loss
            loss.backward() # Backpropagate the loss
            optimizer.step() # Update the model parameters

            train_loss += loss.item() # Accumulate the training loss
            _, predicted = outputs.max(1) # Get the predictions
            total += labels.size(0) # Accumulate the number of examples
            correct += predicted.eq(labels).sum().item() # Accumulate the number of correct predictions
        
        train_acc = 100 * correct / total # Calculate the training accuracy
        train_loss /= len(train_loader) # Average the training loss

        # Initialize lists to store the metrics
        model.eval() # Set the model to evaluation mode
        test_loss = 0
        correct = 0
        total = 0
        black_holes_correct = 0
        sphalerons_correct = 0
        black_holes_total = 0
        sphalerons_total = 0
        all_preds = []
        all_labels = []

        with torch.no_grad(): # Disable gradient computation 
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device).float(), labels.to(device).long() # Move inputs and labels to device
                outputs = model(inputs).float() # Pass inputs to the model and get the outputs
                
                loss = criterion(outputs, labels) # Calculate the loss
                test_loss += loss.item()
                _, predicted = outputs.max(1) # Get the predictions
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
        black_holes_precision, sphalerons_precision = precision[0], precision[1]
        black_holes_recall, sphalerons_recall = recall[0], recall[1]

        # Append the metrics to the lists
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        test_losses.append(test_loss)
        test_accs.append(test_acc)
        black_holes_accs.append(black_holes_acc)
        sphalerons_accs.append(sphalerons_acc)
        test_precisions.append(precision)
        test_recalls.append(recall)

        # # Print results
        # print(f"Epoch: {epoch}/{n_epochs}")
        # print(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.2f}%")
        # print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.2f}%")


        metrics = {
            'epoch': epoch,
            'train_losses': train_loss,
            'train_accs': train_acc,
            'test_losses': test_loss,
            'test_accs': test_acc,
            'black_holes_accs': black_holes_acc,
            'sphalerons_accs': sphalerons_acc,
            'precisions': precision,
            'recalls': recall,
            'all_preds': all_preds,
            'all_labels': all_labels
        }
        metrics_per_epoch.append(metrics)
    
    return metrics_per_epoch