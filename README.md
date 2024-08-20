# Predicting functional groups from IR Spectra
This repository contains scripts to train a convolutional neural network (CNN) to predict the functional groups present in an IR spectrum. I.e., for a set of wavelengths and corresponding absorbance values, the model should predict the functional groups present. 

## How to Use
1. Download MLKoji IR data and place in folder '0'.
2. Create a virtual environment and install dependencies:<br>
   $ pip install -r requirements.txt<br>
4. Run files in this order:<br>
   a. split_data.py<br>
   b. hyperparameter_optimization.py<br>
   c. train_model.py (update based on results of hyperparameter optimization)<br>
   d. optimal_thresholding.py<br>
   e. evaluation.py<br>  

## Acknowledgments
This project includes the following open-source components:  
- **Algorithm for identifying functional groups from SMILES in ifg.py**: Adapted from Richard Hall and Guillaume Godin in RDKit's repository (https://github.com/rdkit/rdkit/blob/master/Contrib/IFG/ifg.py)
- **CNN model architecture in hyperparameter_optimization.py**: Adapted from https://github.com/gj475/irchracterizationcnn/tree/main<br>
  
The IR spectra used was taken by Koji Nakanishi.
