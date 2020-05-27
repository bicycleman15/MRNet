from dataset import MRData
from models import MRnet
from config import config
import torch
from utils import _get_trainable_params, _run_eval

"""Performs training of a specified model.
    
Input params:
    config_file: Takes in configurations to train with 
"""

def train(config : dict, export=True):
    print('Starting to Train Model...')

    print('Loading Train Dataset...')
    train_data = MRData(task='acl',train=True)
    train_loader = data.DataLoader(
        train_data, batch_size=1, num_workers=1, shuffle=True
    )

    print('Loading Validation Dataset...')
    val_data = MRData(task='acl',train=True)
    val_loader = data.DataLoader(
        val_data, batch_size=1, num_workers=1, shuffle=False
    )

    print('Initializing Model...')
    model = MRnet()

    print('Initializing Loss Method...')
    # TODO : maybe take a wiegthed loss
    criterion = torch.nn.CrossEntropyLoss()

    print('Setup the Optimizer')
    # TODO : Add other hyperparams as well
    optimizer = optim.Adam(_get_trainable_params(model), lr=config['lr'])

    starting_epoch = config['starting_epoch']
    num_epochs = config['max_epoch']

    best_accuracy = 0.0

    # TODO : add tqdm with support with notebook
    for epoch in range(starting_epoch, num_epochs):

        epoch_start_time = time.time()  # timer for entire epoch

        train_iterations = len(train_dataset)
        train_batch_size = config['batch_size']

        num_batch = 0

        # loss for the epoch
        total_loss = 0.0

        model.train()

        # TODO : add tqdm here as well ? or time remaining ?
        for batch in train_loader:

            images, label = batch

            # TODO: Add some visualiser maybe

            output = model.forward(images)

            # Calculate Loss cross Entropy
            loss = criterion(output, label)
            # TODO : Add loss in TensorBoard

            # add loss to epoch loss
            total_loss += loss.item()

            # Do backpropogation
            loss.backward()

            # Change wieghts
            optimizer.step()

            # zero out all grads
            criterion.zero_grad()
            optimizer.zero_grad()

            # Log some info, TODO : add some graphs after some interval
            if num_batch % config['log_freq']:
                print('{}/{} Epoch : {}/{} Batch Iter : Batch Loss {}'.format(
                    epoch, num_epochs, num_batch, len(train_loader), loss.item()
                ))

            num_batch += 1
        
        # Set to eval mode
        model.eval()
        
        # Print details about end of epoch
        validation_loss, accuracy = _run_eval(model, val_loader)

        # TODO : Print details about end of epoch
        # Accuracy, Train Loss, Val Loss, Learning Rate

        if best_accuracy < accuracy :
            # Save this model
            
            best_accuracy = accuracy
        
        # TODO : Change LR depending upon epoch, LR

        total_loss = 0.0
        print('End of epoch {0} / {1} \t Time Taken: {2} sec'.format(epoch, num_epochs, time.time() - epoch_start_time))

    if export:
        print('Exporting model...')
        # TODO : save model to disk


if __name__ == '__main__':

    print('Training Configuration')
    print(config)

    train()

    print('Training Ended...')






