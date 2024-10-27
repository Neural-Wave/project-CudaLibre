import torch
import time
from tqdm import tqdm
from sklearn.metrics import fbeta_score

def train_model(model, 
                train_loader, 
                val_loader, 
                criterion, 
                optimizer, 
                num_epochs, 
                writer, 
                device, 
                checkpoint_path, 
                beta=0.5):
    
    model.train() 

    for epoch in range(num_epochs):
        running_loss = 0.0
        start_time = time.time()

        # Store all labels and predictions for F-beta calculation
        all_labels = []
        all_predictions = []

        # Training Loop
        for images, labels in tqdm(train_loader):
            images, labels = images.to(device), labels.to(device)
            labels = labels.unsqueeze(1).to(torch.float32)
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Compute loss
            running_loss += loss.item()

            # Store labels and predictions for F-beta score calculation
            predicted = (outputs > 0.5).float()  # Threshold outputs for binary classification
            all_labels.extend(labels.cpu().detach().numpy().astype(int))
            all_predictions.extend(predicted.cpu().detach().numpy().astype(int))

        # Compute training metrics
        epoch_loss = running_loss / len(train_loader)

        # Compute F-beta score for training
        train_fbeta = fbeta_score(all_labels, all_predictions, beta=beta, average='binary')

        epoch_time = time.time() - start_time

        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}, F-beta Score: {train_fbeta:.4f}, Time: {epoch_time:.2f}s")

        # Log training metrics to TensorBoard
        writer.add_scalar('Training/Loss', epoch_loss, epoch)
        writer.add_scalar('Training/F-beta Score', train_fbeta, epoch)

        # Validation Loop
        model.eval()  # Set the model to evaluation mode
        val_running_loss = 0.0
        val_all_labels = []  # Reset validation labels list at each epoch
        val_all_predictions = []  # Reset validation predictions list at each epoch

        with torch.no_grad():  # Disable gradient calculation
            for val_images, val_labels in val_loader:
                val_images, val_labels = val_images.to(device), val_labels.to(device)
                val_labels = val_labels.unsqueeze(1).to(torch.float32)

                val_outputs = model(val_images)
                val_loss = criterion(val_outputs, val_labels)

                val_running_loss += val_loss.item()

                # Correctly use val_outputs for predictions in validation
                val_predicted = (val_outputs > 0.5).float()  # Threshold val_outputs for binary classification
                val_all_labels.extend(val_labels.cpu().detach().numpy().astype(int))
                val_all_predictions.extend(val_predicted.cpu().detach().numpy().astype(int))

        # Compute validation metrics
        val_epoch_loss = val_running_loss / len(val_loader)

        # Ensure the lists have consistent lengths before calculating F-beta score
        assert len(val_all_labels) == len(val_all_predictions), "Mismatch in length of labels and predictions in validation"

        # Compute F-beta score for validation
        val_fbeta = fbeta_score(val_all_labels, val_all_predictions, beta=beta, average='binary')

        # Log validation metrics to TensorBoard
        writer.add_scalar('Validation/Loss', val_epoch_loss, epoch)
        writer.add_scalar('Validation/F-beta Score', val_fbeta, epoch)

        print(f"Validation Loss: {val_epoch_loss:.4f}, F-beta Score: {val_fbeta:.4f}")

        # Save the model checkpoint every 5 epochs
        if (epoch + 1) % 5 == 0 and checkpoint_path is not None:
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': epoch_loss,  # Optionally save the loss
            }
            torch.save(checkpoint, f"{checkpoint_path}/model_epoch_{epoch + 1}.pt")
            print(f"Checkpoint saved at epoch {epoch + 1}")
