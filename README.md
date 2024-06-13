# Predicting functional groups from IR Spectra
Scripts for data cleaning, normalization, and feature extraction of IR data taken from 100 different compounds by Koji Nakanishi.

This repository contains scripts to train a neural network to predict the functional groups present in an IR spectrum. I.e., for a set of wavelengths and corresponding absorbance values, the model should predict the functional groups present. 

To train the model, data from the IR spectra of 100 different compounds was taken. The data was preprocessed to normalize the absorbances and get all the wavelengths that the compounds were measured at. A predictor table was made to define the predictor classes. 

For each compound, the absorbance values at each wavelength and the 

## How to Use
1. Clone this repository to your local machine.
2. Download MLKoji IR data. Navigate to MLKoji directory. Open NIST_IR_0.zip file. Make new folders listed in directory tree.
   
Directory tree:</br>
.</br>
├── 0</br>
├── 0_test100</br>
    ├── casnos.txt</br>
    ├── compounds.txt</br>
    ├── get_IRdata.py</br>
    ├── ifg.py</br>
    ├── normalize_absorbances.py</br>
    ├── normalized_absorbances</br>
    ├── normalized_absorbances_with_defects</br>
    ├── write_smiles.py</br>
    └── train_nn.m

4. Run get_IRdata.py. 
5. Run train_nn.m.

## Dependencies
- Python 3.0 or above
- RDKit
- PubChem
- MATLAB
  
## Acknowledgments
This project includes the following open-source components:  
- **Algorithm for identifying functional groups from SMILES in ifg.py**: Adapted from Richard Hall and Guillaume Godin in RDKit's repository (https://github.com/rdkit/rdkit/blob/master/Contrib/IFG/ifg.py)

The IR spectra used was taken by Koji Nakanishi.
## License
This project is licensed under the BSD 3-Clause License - see the LICENSE file for details.
