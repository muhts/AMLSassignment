"""
TRAINS models
"""

import torch
from torchvision import transforms, utils
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import csv
import os
from sklearn.model_selection import train_test_split
from classes.data_loader import split_dataloader, readFormat,FacesDataset, RandomCrop
from classes.Models import get_alexnet, get_resnet18, test_model, train_model
from classes.SimpleCNN import EmotionCNN, AgeCNN, GlassesCNN, HumanCNN, HairColorCNN
from classes.data_cleaner import removeNeg, replaceNeg, cleanFaces

#ignore warnings
import warnings
warnings.filterwarnings("ignore")

if __name__ == "__main__":
    print('### Trains Models ###')
    # label selection
    print()
    print('label choices: ')
    print('Task 1: smiling ' )
    print('Task 2: young ')
    print('Task 3: eyeglasses ')
    print('Task 4: human ')
    print('Task 5: hair_color')
    print()
    print('e.g Enter smiling for task 1')
    print()

    label = input('Feature to classify: ' )

    # reads dataset
    loader = split_dataloader("attribute_list.csv",label)

    # model selection
    print()
    print('Model choices: ')
    print('alexnet')
    print('resnet18')
    print('simple (SimpleCNN)')
    print()
    modelname = input('Which model?: ' )
    #load resnet18
    if modelname == 'alexnet':
        model,epochs = get_alexnet(len(loader['classes']))
    #load resnet18
    elif modelname == 'resnet18':
        model,epochs = get_resnet18(len(loader['classes']))

    # Load respective tuned models for simpleCNN
    elif modelname == 'simple':
        if label == 'smiling':
            model = EmotionCNN
            epochs = 45
        elif label == 'young':
            model = AgeCNN
            epochs = 35
        elif label == 'eyeglasses':
            model = GlassesCNN
            epochs = 55
        elif label == 'human':
            model = HumanCNN
            epochs = 40
        elif label == 'hair_color':
            model = HairColorCNN
            epochs = 50

    # get save name
    save_name = label+'_'+modelname

    weights = os.path.isfile('savedmodels/'+save_name+'.pt')
    if weights:
        print('Weights have been found for the model')
        training_ = input('Still wish to train? (y,[n]): ' )
    else:
        training_ ='y'

    if training_ =='y':
         # load training dataset
         loader = split_dataloader("attribute_list.csv",label)
         # train model
         print('Weights not trained. Training will begin....')
         z = train_model(loader['train'],loader['val'],model,n_epochs=epochs,save_file=save_name)
         test_loader = loader['test']
    else:
        # testing dataset
        print()
        print('### Test ###')
        csv_ = input('Enter name of attribute file to test: ')
        folder_ = input('Enter dataset location: ')

        # Converting to DataLoader
        df = readFormat(csv_)
        df = cleanFaces(df)
        if label == 'hair_color':
            df = removeNeg(df)
        else:
            df = replaceNeg(df)

        test_data = FacesDataset(df['file_name'],df[label],root_dir=folder_,transform=transforms.Compose([
                                                                            RandomCrop(224),
                                                                            transforms.ToTensor()
                                                                           ]))
        test_loader = DataLoader(test_data, batch_size=1, shuffle=True)

    # test model
    model.load_state_dict(torch.load('savedmodels/'+save_name+'.pt'))
    load = test_model(test_loader,model)


    save_ = input('save file? (y,[n]): ')
    if save_ =='y':
        # Write to csv
        l_dict = {
            'file_name':load['file_name'],
            'prediction':load['prediction']
        }
        l = pd.DataFrame(l_dict)
        file = open(label+'_'+modelname +'.csv','w')
        out = csv.writer(file)

        out.writerows([[str(load['accuracy'])[:5]]])
        for i in range(len(load['file_name'])):
            pred = load['prediction'][i]
            name = load['file_name'][i]
            out.writerow(l.iloc[i])

        file.close()
