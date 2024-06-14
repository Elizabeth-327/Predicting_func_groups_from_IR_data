# Predicting functional groups from IR Spectra
This repository contains scripts to train a neural network to predict the functional groups present in an IR spectrum. I.e., for a set of wavelengths and corresponding absorbance values, the model should predict the functional groups present. 

To train the model, data was taken from the IR spectra of 100 different compounds. The data was preprocessed to normalize the absorbances and get all the wavelengths that the compounds were measured at. The functional groups in each compound were found by using an RDKit-based algorithm written by Richard Hall and Guillame Godin, credited in the Acknowledgments section. 

Part of the predictor table:
<img width="902" alt="Screenshot 2024-06-14 at 11 14 33 AM" src="https://github.com/Elizabeth-327/Predicting_func_groups_from_IR_data/assets/118557290/08549558-cc58-41dc-b6ed-cd9b0fe4df19"></br>
Each row corresponds to one compound. "None" indicates that no absorbance value was recorded for that specific wavelength. Each column represents one predictor class. There are 81721 wavelengths in the complete predictor table. 

Part of the response table:
<img width="842" alt="Screenshot 2024-06-13 at 4 45 10 PM" src="https://github.com/Elizabeth-327/Predicting_func_groups_from_IR_data/assets/118557290/cfa9baea-998c-4363-ae85-a7e78d9fc3f4"></br>
Each row corresponds to one compound. '0' indicates the lack of the functional group in the compound; '1' indicates the presence of the functional group. Each column represents one response class. 

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
