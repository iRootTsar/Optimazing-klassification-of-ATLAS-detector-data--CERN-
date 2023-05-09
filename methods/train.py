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

def train(model, train_loader, test_loader, optimizer, criterion, n_epochs, scheduler):
    
    #save best model
    best_val_loss = float("inf")  # Initialize the best validation loss as infinity
    best_model_state = None  # Initialize the best model state as None

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
        best_all_preds = [] # Initialize the best all_preds list
        best_all_labels = []  # Initialize the best all_labels list
    
    
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
        val_loss = 0

        with torch.no_grad(): # Disable gradient computation 
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device).float(), labels.to(device).long() # Move inputs and labels to device
                outputs = model(inputs).float() # Pass inputs to the model and get the outputs
                
                val_loss += loss.item()
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
        
        if test_loss < best_val_loss:  # If the current validation loss is lower than the best validation loss
            best_val_loss = test_loss  # Update the best validation loss
            best_model_state = model.state_dict()  # Save the current model state
            best_all_preds = all_preds
            best_all_labels = all_labels

        scheduler.step(val_loss) #Update learning rate using the scheduler
        
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
    
    return metrics_per_epoch, best_model_state, best_all_preds, best_all_labels