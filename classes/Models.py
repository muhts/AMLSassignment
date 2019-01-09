#import libraries
import numpy as np
import torch
from torch import nn, optim
import torch.nn.functional as F
import torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils, models
import time

def get_alexnet(labels=2):
    """
    loads alexnet and downloads its weights
    """
    # load model
    model = models.alexnet(pretrained=True)
    num_ftrs = model.classifier[6].in_features
    # replace output layer
    model.classifier[6] = nn.Linear(num_ftrs, labels)
    return model, 40 #model and epochs

def get_resnet18(labels=2):
    """
    loads resnet18 and downloads its weights
    """
    # load model
    model = models.resnet18(pretrained=True)
    num_ftrs = model.fc.in_features
    # replace output layer
    model.fc = nn.Linear(num_ftrs, labels)
    return model, 40 #model and epochs

def test_model(test_loader,model):
    """
    Test Accuracy

    Args:
        test_loader = dataloader object containing Dataset
        model = pytorch model being tested
    """
    file_name = []
    prediction = []
    train_on_gpu = torch.cuda.is_available()
    if train_on_gpu:
        print('using gpu')
        model.cuda()

    with torch.no_grad():
        correct = 0
        total = 0
        for data , target , name in test_loader:
            #train on gpu if present
            if train_on_gpu:
                data, target = data.cuda(), target.cuda()
            outputs = model(data)

            predicted = torch.argmax(outputs,dim=1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
            fname = name.cpu().numpy()
            pred = predicted.cpu().numpy()
            file_name.append(str(fname[0])+'.png')
            prediction.append(pred[0])

            testAccuracy = 100 * correct / total
        print('Test Accuracy: {} %'.format(testAccuracy))

    return {'accuracy':testAccuracy,
            'file_name':file_name,
            'prediction':prediction}



def predict(pred_loader,model):
    """
    Test Accuracy

    Args:
        pred_loader = dataloader object containing Dataset
        model = pytorch model being tested
    """
    file_name = []
    prediction = []
    train_on_gpu = torch.cuda.is_available()
    if train_on_gpu:
        print('using gpu')
        model.cuda()

    with torch.no_grad():
        correct = 0
        total = 0
        for data , target , name in pred_loader:
            #train on gpu if present
            if train_on_gpu:
                data, target = data.cuda(), target.cuda()
            outputs = model(data)
            predicted = torch.argmax(outputs,dim=1)

            fname = name.cpu().numpy()
            pred = predicted.cpu().numpy()

            file_name.append(str(fname[0])+'.png')
            prediction.append(pred[0])


    return {'file_name':file_name,
            'prediction':prediction}

def train_model(train_loader,val_loader, model,n_epochs = 50,save_file=None):
    """
    Trains model

    Args:
        train_loader = training Dataset
        val_loader = validation dataset
        model = pytorch model
        n_epochs = numer of n_epochs
        save_file = name of save file

    Output = dict containing the model, train, validation losses and accuracy
    """
    train_on_gpu = torch.cuda.is_available()
    if train_on_gpu:
        print('using gpu')
        model.cuda()
    # specify loss function (categorical cross-entropy)
    criterion = nn.CrossEntropyLoss()
    # specify optimizer
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    # empyt
    t_loss = []
    v_loss = []
    accuracy = []

    valid_loss_min = np.Inf # track change in validation loss

    for epoch in range(1, n_epochs+1):
        start_time = time.time()
        # keep track of training and validation loss
        train_loss = 0.0
        valid_loss = 0.0


        model.train()
        for data, target , name in train_loader:
            if train_on_gpu:
                data, target = data.cuda(), target.cuda()

            optimizer.zero_grad()

            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()*data.size(0)


        model.eval()
        for data, target, name in val_loader:
            if train_on_gpu:
                data, target = data.cuda(), target.cuda()

            output = model(data)
            loss = criterion(output, target)
            valid_loss += loss.item()*data.size(0)

        # calculate average losses
        train_loss = train_loss/len(train_loader.dataset)
        valid_loss = valid_loss/len(val_loader.dataset)

        # print training/validation loss
        print('Epoch: {} \tTrain Loss: {:.6f} \tVal Loss: {:.6f}'.format(
            epoch, train_loss, valid_loss))
        timeTaken = time.time() - start_time

        # print validation accuracy
        with torch.no_grad():
            correct = 0
            total = 0
            for data , target, name in val_loader:
                if train_on_gpu:
                    data, target = data.cuda(), target.cuda()
                outputs = model(data)
                predicted = torch.argmax(outputs,dim=1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
            print('Validation Accuracy: {} %'.format(100 * correct / total))
        print('Time taken: '+ str(timeTaken))


        # save model if validation loss has decreased
        if save_file != None:
            if valid_loss <= valid_loss_min:
                print('Saving model ...')
                torch.save(model.state_dict(), 'savedmodels/'+save_file+'.pt')
                valid_loss_min = valid_loss
        print()
        t_loss.append(train_loss)
        v_loss.append(valid_loss)
        accuracy.append(100 * correct / total)

    return {'model' : model,
            'val_loss' : v_loss,
            'train_loss' : t_loss,
            'accuracy' : accuracy}
