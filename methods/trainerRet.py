import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support

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

def train(model, train_loader, test_loader, optimizer, criterion, n_epochs, scheduler):
    
    # Initialize lists to store the metrics
    train_losses = []
    train_accs = []
    test_losses = []
    test_accs = []
    black_holes_accs = []
    sphalerons_accs = []
    black_holes_precisions = []
    sphalerons_precisions = []
    black_holes_recalls = []
    sphalerons_recalls = []
    black_holes_f1_scores = []
    sphalerons_f1_scores = []
    test_preds_and_labels = []
    
    
    for epoch in range(n_epochs):
        # Train
        model.train() #set the model to train mode
        train_loss = 0
        correct = 0
        total = 0
    
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device).float(), labels.to(device).long() #Mode inputs and labels to the device
            optimizer.zero_grad() #Zero the gradients
            outputs = model(inputs).float() #Pass inputs to the model and get the outputs
            loss = criterion(outputs, labels) #calculate the loss
            loss.backward() #Backpropogate the loss
            optimizer.step() #Update the model parameters

            train_loss += loss.item() #Accumulate the trainings loss
            _, predicted = outputs.max(1) #Get the predictions
            total += labels.size(0) #Accumulate the number of examples
            correct += predicted.eq(labels).sum().item() #Accumulate the number of correct predictions
        
        train_acc = 100 * correct / total #Caulculate the training accuracy
        train_loss /= len(train_loader) #Average the training loss

        # Test
        model.eval() #Set the model to evaluation mode
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

        with torch.no_grad(): #Disable gradient computation 
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device).float(), labels.to(device).long() #Move inputs and labels to device
                outputs = model(inputs).float() #Pass inputs to the model and get the outputs
                
                loss = criterion(outputs, labels) #Calculate the loss

                val_loss += loss.item()
                test_loss += loss.item()
                _, predicted = outputs.max(1) #Get the predictions
                total += labels.size(0) #Accumulate the number of examples
                correct += predicted.eq(labels).sum().item() #Accumulate the number of correct predictions

                # Separate accuracies for black holes and sphalerons
                black_holes_correct += (predicted * labels).sum().item()
                sphalerons_correct += ((1 - predicted) * (1 - labels)).sum().item()
                black_holes_total += labels.sum().item()
                sphalerons_total += (1 - labels).sum().item()

                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                
                test_preds_and_labels.append((outputs, labels))
        
        test_acc = 100 * correct / total
        black_holes_acc = 100 * black_holes_correct / black_holes_total
        sphalerons_acc = 100 * sphalerons_correct / sphalerons_total
        test_loss /= len(test_loader)
        
        # Calculate precision and recall
        precision, recall, f1_scores, _ = precision_recall_fscore_support(all_labels, all_preds, average=None)
        black_holes_precision, sphalerons_precision = precision
        black_holes_recall, sphalerons_recall = recall
        black_holes_f1_score, sphalerons_f1_score = f1_scores

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
        black_holes_precisions.append(black_holes_precision)
        sphalerons_precisions.append(sphalerons_precision)
        black_holes_recalls.append(black_holes_recall)
        sphalerons_recalls.append(sphalerons_recall)
        black_holes_f1_scores.append(black_holes_f1_score)
        sphalerons_f1_scores.append(sphalerons_f1_score)

        scheduler.step(val_loss) #Update learning rate using the scheduler
    
    # Return the collected metrics
    metrics = {
        'train_losses': train_losses,
        'train_accs': train_accs,
        'test_losses': test_losses,
        'test_accs': test_accs,
        'black_holes_accs': black_holes_accs,
        'sphalerons_accs': sphalerons_accs,
        'black_holes_precisions': black_holes_precisions,
        'sphalerons_precisions': sphalerons_precisions,
        'black_holes_recalls': black_holes_recalls,
        'sphalerons_recalls': sphalerons_recalls,
        'black_holes_f1_scores': black_holes_f1_scores,
        'sphalerons_f1_scores': sphalerons_f1_scores,
        'preds_and_labels': test_preds_and_labels, # Add this line
    }
    
    return metrics