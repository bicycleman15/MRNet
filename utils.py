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

    i = 0
    for images, label in validation_loader:

        images = [x.cuda() for x in images]
        label = label.cuda()

        with torch.no_grad():
            output = model.forward(images)

            # Calc loss
            loss = criterion(output, label)

            # TODO : label in form of [[1,0]] whereas output in [0.96,0.04], dimension do not match
            if torch.argmax(output).item() == label[0].item():
                # print('Correct !!')
                correct_cases += 1
            
            validation_loss += loss.item()
            
            print('Validation Loss is {} at iter {}/{}'.format(loss.item(), i, len(validation_loader)))
            
        
        i += 1

    # Calculate accuracy
    accuracy = float(correct_cases) / len(validation_loader)

    # Calculate Loss per patient
    average_val_loss = validation_loss / len(validation_loader)

    print('Accuracy is {}'.format(accuracy))

    return average_val_loss, accuracy
