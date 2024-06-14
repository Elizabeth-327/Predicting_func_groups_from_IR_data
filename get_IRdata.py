import os
import re
import shutil
from normalize_absorbances import normalize_ir_spectrum
from decimal import Decimal
from ifg import get_func_groups_from_compound_name
import scipy.io
import numpy as np

def get_sibling_directory_path(sibling_dir_name):
    #Get the directory of the current script
    script_dir = os.path.dirname(__file__)

    #Construct the path to the sibling directory
    sibling_dir = os.path.join(script_dir, '..', sibling_dir_name)

    #Normalize the path to get the absolute path
    sibling_dir = os.path.abspath(sibling_dir)

    return sibling_dir

def get_specific_filenames(casnos_txt):

    filenames = []
    with open(casnos_txt, 'r') as file:

        for line in file:
            casno = line.strip()
            casno += '.jdx'
            filenames.append(casno)

    return set(filenames) #convert list to set for faster lookup


def list_files_in_directory(directory):
    #List all .jdx files in the given directory
    return [f for f in os.listdir(directory) if f.endswith('.jdx')] #returns a list of string names of the files

def filter_files(all_files, specific_filenames):
    #Filter out the specific files we want
    return [f for f in all_files if f in specific_filenames] #returns a list of string names of the files

def normalize(input_directory, input_files, output_directory):

    for input_filename in input_files:
        input_file_path = os.path.join(input_directory, input_filename)
        
        output_filename = input_filename.split('.')[0] + '_normalized.' + 'jdx'
        output_file_path = os.path.join(output_directory, output_filename)
        
        try:
            with open(input_file_path, 'r') as input_file:
                normalize_ir_spectrum(input_file_path, output_file_path) 
        except FileNotFoundError:
            print(f'File {input_filename} not found.')

def remove_defective_files(input_directory, input_files, destination_directory):

    #clear the destination directory
    if os.path.exists(destination_directory):
        shutil.rmtree(destination_directory)
    os.makedirs(destination_directory)
    
    defective_files = []
    for input_filename in input_files:
        input_file_path = os.path.join(input_directory, input_filename)

        with open(input_file_path, 'r') as input_file:
            lines = input_file.readlines()

        defective = True
        for line in lines:
            line = line.strip()

            if re.match(r'^[\d\s.]+$', line):
                defective = False
                break

        if defective:
            defective_files.append(input_filename)
        else:
            shutil.copy(input_file_path, destination_directory)
    
    return defective_files


def get_all_wavelengths(input_directory, input_files): 

    wavelengths = []
    for input_filename in input_files:
        input_file_path = os.path.join(input_directory, input_filename)


        with open(input_file_path, 'r') as input_file:
            lines = input_file.readlines()

        for line in lines:
            line = line.strip()

            if line.startswith('##DELTAX'):
                deltax = Decimal(line.split('=')[1])

            elif re.match(r'^[\d\s.]+$', line):

                data = line.split(' ')
                first_wavelength = Decimal(data[0])             
                wavelengths.append(first_wavelength)

                for i in range(2, len(data)):
                    wavelength = first_wavelength + (i-1)*deltax
                    wavelengths.append(wavelength)

    wavelengths = set(wavelengths)
    wavelengths = list(wavelengths)
    wavelengths.sort()
    return wavelengths
            

def match_absorbances_to_wavelengths(input_directory, input_files):

    all_wavelengths_absorbances = []
    
    for input_filename in input_files:
        input_file_path = os.path.join(input_directory, input_filename)
        
        with open(input_file_path, 'r') as input_file:
            lines = input_file.readlines()

        wavelengths_absorbances = {}
        for line in lines:
            line = line.strip()

            if line.startswith('##DELTAX'):
                deltax = Decimal(line.split('=')[1])
            
            elif re.match(r'^[\d\s.]+$', line):

                data = line.split(' ')
                
                first_wavelength = float(data[0])
                first_wavelength = Decimal(str(first_wavelength))

                first_absorbance = float(data[1])
                wavelengths_absorbances.update({first_wavelength: first_absorbance})
                
                for i in range(2, len(data)):
                    wavelength = first_wavelength + (i-1)*deltax
                    absorbance = float(data[i])
                    wavelengths_absorbances.update({wavelength: absorbance})

        all_wavelengths_absorbances.append((input_filename, wavelengths_absorbances))

    return all_wavelengths_absorbances

def add_nones(all_wavelengths_absorbances, all_wavelengths):

    updated_all_wavelengths_absorbances = []
    
    for (input_filename, wavelength_absorbances) in all_wavelengths_absorbances:
        wavelengths = wavelength_absorbances.keys()

        for wavelength in all_wavelengths:
            if wavelength not in wavelengths:
                wavelength_absorbances.update({wavelength: 'None'})

        sorted_wavelength_absorbances = dict(sorted(wavelength_absorbances.items()))
        updated_all_wavelengths_absorbances.append((input_filename, sorted_wavelength_absorbances))

    return updated_all_wavelengths_absorbances

def construct_final_data(all_wavelengths_absorbances, casnos_txt, compounds_txt):

    casnos = []
    with open(casnos_txt, 'r') as casnos_txt_file:
        for line in casnos_txt_file:
            casno = line.strip()
            casnos.append(casno)

    compounds = []
    with open(compounds_txt, 'r') as compounds_txt_file:
        for line in compounds_txt_file:
            compound_name = line.strip()
            compounds.append(compound_name)

    all_func_groups = []
    for compound_name in compounds:
        func_groups = get_func_groups_from_compound_name(compound_name)
        all_func_groups.append(func_groups)
        
    data = []
    for (input_filename, wavelengths_absorbances) in all_wavelengths_absorbances:
        casno = input_filename.split('_')[0]

        #find corresponding functional groups
        for i in range(len(casnos)):
            if casnos[i] == casno:
                compound_func_groups = all_func_groups[i]

        #extract absorbances of the compound
        absorbances = []
        for absorbance in wavelengths_absorbances.values():
            absorbances.append(absorbance)

        #create absorbance-functional_group dictionary for the compound
        func_groups_absorbances = {'absorbances': absorbances, 'functional groups': compound_func_groups}
        
        data.append(func_groups_absorbances)

    return data

    
def main():
    
    #get absolute path for '0' directory 
    _0_dir = get_sibling_directory_path('0')

    #get a list of all the jdx files in '0'
    all_files = list_files_in_directory(_0_dir)

    #get the specific 100 files we want from '0' 
    specific_filenames = get_specific_filenames('casnos.txt')
    files_to_read = filter_files(all_files, specific_filenames)

    #normalize absorbances in the 100 files
    normalize(_0_dir, files_to_read, 'normalized_absorbances')

    #get non-defective files
    normalized_absorbances_with_defects_dir = os.path.abspath(os.path.join(os.getcwd(), 'normalized_absorbances_with_defects'))
    normalized_absorbances_dir = os.path.abspath(os.path.join(os.getcwd(), 'normalized_absorbances'))
    remove_defective_files(normalized_absorbances_with_defects_dir, list_files_in_directory(normalized_absorbances_with_defects_dir), normalized_absorbances_dir)

    non_defective_files = list_files_in_directory(normalized_absorbances_dir)
    all_wavelengths = get_all_wavelengths(normalized_absorbances_dir, non_defective_files)

    with open('all_wavelengths.txt', 'w') as file:
        
        for wavelength in all_wavelengths:
            file.write(f'{wavelength}\n')


    all_wavelengths_absorbances = match_absorbances_to_wavelengths(normalized_absorbances_dir, non_defective_files)
    all_wavelengths_absorbances = add_nones(all_wavelengths_absorbances, all_wavelengths)

    with open('wavelengths_absorbances.txt', 'w') as file:
        line_count = 0
        for (input_filename, wavelengths_absorbances) in all_wavelengths_absorbances:
            file.write(f'File: {input_filename}\n')
            line_count += 1
            for wavelength, absorbance in wavelengths_absorbances.items():
                file.write(f'Wavelength: {wavelength} Absorbance: {absorbance}\n')
                line_count += 1
    print('line count =', line_count)

    
    data = construct_final_data(all_wavelengths_absorbances, 'casnos.txt', 'compounds.txt')

    with open('data.txt', 'w') as file:
        for elem in data:
            file.write(f'{elem}\n')

    #convert data into numpy arrays to import to MATLAB
    all_absorbances = [sample['absorbances'] for sample in data]

    all_revised_absorbances = []
    for absorbances in all_absorbances:

        revised_absorbances = []
        for i in range(len(absorbances)):
            revised_absorbances.append(absorbances[i])

        all_revised_absorbances.append(revised_absorbances)
        
    all_func_groups = [sample['functional groups'] for sample in data]

    X = np.array(all_revised_absorbances)

    #process functional groups for binary encoding
    all_unique_groups = set()
    for func_groups in all_func_groups:
        for func_group in func_groups:
            all_unique_groups.add(func_group)

    #create a mapping from functional group to index
    group_to_index = {group: idx for idx, group in enumerate(all_unique_groups)}

    #create binary vectors for each compound
    binary_func_groups = []
    for func_groups in all_func_groups:
        binary_vector = [0] * len(all_unique_groups)
        for func_group in func_groups:
            binary_vector[group_to_index[func_group]] = 1
        binary_func_groups.append(binary_vector)

    #convert binary vectors to NumPy array
    y = np.array(binary_func_groups)


    matlab_dir = os.path.expanduser('~/Documents/MATLAB')
    predictor_relative_path = 'zero_test100/predictor_array.mat'
    predictor_mat_file_path = os.path.join(matlab_dir, predictor_relative_path)

    response_relative_path = 'zero_test100/response_array.mat'
    response_mat_file_path = os.path.join(matlab_dir, response_relative_path)
    
    scipy.io.savemat(predictor_mat_file_path, {'predictor_array': X})
    scipy.io.savemat(response_mat_file_path, {'response_array': y})
    
    all_wavelengths = [float(wavelength) for wavelength in all_wavelengths]

    all_wavelengths_array = np.array(all_wavelengths)
    wavelengths_relative_path = 'zero_test100/wavelengths_array.mat'
    wavelengths_mat_file_path = os.path.join(matlab_dir, wavelengths_relative_path)
    scipy.io.savemat(wavelengths_mat_file_path, {'wavelengths_array': all_wavelengths_array})

    all_func_groups_array = np.array(list(all_unique_groups))
    func_groups_relative_path = 'zero_test100/func_groups_array.mat'
    func_groups_mat_file_path = os.path.join(matlab_dir, func_groups_relative_path)
    scipy.io.savemat(func_groups_mat_file_path, {'func_groups_array': all_func_groups_array})

    
main()
