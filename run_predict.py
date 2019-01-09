"""
Predicts models
"""

import torch
import pandas as pd
import csv
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils , models
from classes.data_loader import read_new, FacesDataset, RandomCrop
from classes.Models import get_alexnet, get_resnet18
from classes.Models import test_model, train_model, predict
from classes.SimpleCNN import EmotionCNN, AgeCNN, GlassesCNN, HumanCNN, HairColorCNN

#ignore warnings
import warnings
warnings.filterwarnings("ignore")

if __name__ == "__main__":
    print('### Predict ###')
    # label selection
    loc = input('location of dataset: ' )
    if loc =='':
        loc='testing_dataset/'

    file_name = read_new(loc)
    labels =  np.ones(len(file_name))
    unknown_dict = {'file_name':file_name,'labels':labels}
    unkn = pd.DataFrame(unknown_dict, dtype=int)
    unknData = FacesDataset(unkn.file_name,unkn.labels,root_dir=loc,
                            transform=transforms.Compose([RandomCrop(224),
                                                        transforms.ToTensor()
                                                        ]))
    unkn_loader = DataLoader(unknData,shuffle=True)

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


    label = input('Feature to predict: ' )
    if label == 'hair_color':
        l = 6
    else:
        l = 2

    print()
    print('Model choices: ')
    print('alexnet')
    print('resnet18')
    print('simple (SimpleCNN)')
    print()
    modelname = input('Which model?: ' )
    #load resnet18
    if modelname == 'alexnet':
        model,epochs = get_alexnet(l)
    #load resnet18
    elif modelname == 'resnet18':
        model,epochs = get_resnet18(l)

    # Load respective tuned models for simpleCNN
    elif modelname == 'simple':
        if label == 'smiling':
            model = EmotionCNN
        elif label == 'young':
            model = AgeCNN
        elif label == 'eyeglasses':
            model = GlassesCNN
        elif label == 'human':
            model = HumanCNN
        elif label == 'hair_color':
            model = HairColorCNN

    # get save name
    save_name = label+'_'+modelname
    # test model
    model.load_state_dict(torch.load('savedmodels/'+save_name+'.pt'))
    load = predict(unkn_loader,model)

    # Write to csv
    print('Saving predictions to csv')
    l_dict = {
        'file_name':load['file_name'],
        'prediction':load['prediction']
    }
    l = pd.DataFrame(l_dict)

    file = open(save_name+'_pred'+'.csv','w')
    out = csv.writer(file)
    for i in range(len(load['file_name'])):
        pred = load['prediction'][i]
        name = load['file_name'][i]
        out.writerow(l.iloc[i])

    file.close()
