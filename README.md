# Predicting functional groups from IR Spectra
This repository contains scripts to train a convolutional neural network (CNN) to predict the functional groups present in an IR spectrum. I.e., for a set of wavelengths and corresponding absorbance values, the model should predict the functional groups present. 

## How to Use
1. Download MLKoji IR data and place in folder '0'.
2. Create a virtual environment and install dependencies:
   $ pip install -r requirements.txt
3. Run files in this order:
   a. split_data.py
   b. hyperparameter_optimization.py
   c. train_model.py (update based on results of hyperparameter optimization)
   d. optimal_thresholding.py
   3. evaluation.py

## Acknowledgments
This project includes the following open-source components:  
- **Algorithm for identifying functional groups from SMILES in ifg.py**: Adapted from Richard Hall and Guillaume Godin in RDKit's repository (https://github.com/rdkit/rdkit/blob/master/Contrib/IFG/ifg.py)
- **CNN model architecture in hyperparameter_optimization.py**: Adapted from https://github.com/gj475/irchracterizationcnn/tree/main
The IR spectra used was taken by Koji Nakanishi.
