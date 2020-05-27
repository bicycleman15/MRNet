import torch

def _get_trainable_params(model):
    """Get Parameters with `requires.grad` set to `True`"""
    trainable_params = []
    for x in model.parameters():
        trainable_params.append(x)
    return trainable_params

def _run_eval(model, validation_loader, criterion):
    """Runs model over val dataset and returns accuracy and avg val loss"""

    print('Running Validation of Model...')

    model.eval()

    correct_cases = 0
    validation_loss = 0.0

    for images, label in validation_loader:

        with torch.no_grad:
            output = model.forward(images)

            # Calc loss
            loss = criterion(output, label)

            # TODO : label in form of [[1,0]] whereas output in [0.96,0.04], dimension do not match
            if torch.argmax(output) == torch.argmax(label):
                correct_cases += 1
            
            validation_loss += loss.item()

    # Calculate accuracy
    accuracy = float(correct_cases) / len(validation_loader)

    # Calculate Loss per patient
    average_val_loss = validation_loss / len(validation_loader)

    return average_val_loss, accuracy
