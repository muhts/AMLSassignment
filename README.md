# AMLSassignment

Libraries used:
Numpy
Pandas
Matplotlib
Scikit-images
OS
glob

Torch
Torchvision
Scikit-learn

#### Training and testing models ###
Run the command: python run_training.py

It will ask for features to classify.
Options are: (don't include quotes)
    'smiling'
    'eyeglasses'
    'young'
    'human'
    'hair_color'

It will also ask for the model to train on.
(transfer learning models will download pre-trained weights)
Options are: (don't include quotes)
    'alexnet' [all Task csvs use this]
    'resnet18'
    'simple'

If the model hasn't been trained then it will go to train the model.
trained weights will be saved as <label><modelName>.pt in the savedmodels folder

If there is a model already trained it will prompt either to retrain.
Input y to retrain.

Otherwise it will prompt to test:
Input the name of the csv containing attribues. (e.g. 'attribute_list.csv')
Input location of dataset images. (e.g. 'testing_dataset/' , include '/')

The trained model or loaded model will be tested and an accuracy output.

Program will prompt to save file. Input y to save.
This will create a csv named: <label><modelName>.csv

### Prediction ###
(only for unknown attributes, will not run without saved trained weights)

Run the command: python run_predict.py

It will ask for location on images to classify.
(submit folder name, Default set to testing_dataset)

It will ask for label and model to predict and use.
(submit names without quotes).

a prediction csv will be created with filenames and predictions
