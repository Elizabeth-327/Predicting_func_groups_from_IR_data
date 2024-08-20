import os
import re
import shutil
from interpolate_data import interpolate_data
from itertools import zip_longest
from decimal import Decimal
from ifg import get_func_groups_from_compound_name

import pandas as pd
import csv
import scipy.io
import numpy as np
import concurrent.futures


# Define paths
casnos_txt = 'casnos.txt'
compounds_txt = 'compounds.txt'
nist_jdx_path = '../0' #directory that has nist_jdx files
interpolated_dir = './interpolated_data' #make this directory if you haven't already

def process_files():

    #get a list of all the jdx files in '0'
    all_nist_filenames = [f for f in os.listdir(nist_jdx_path) if f.endswith('.jdx')] #returns a list of string names of the files in 0

    #get the specific 100 files we want from '0'
    selected_filenames = []
    with open(casnos_txt, 'r') as file:
        for line in file:
            casno = line.strip()
            casno += '.jdx'
            selected_filenames.append(casno)
    selected_filenames = set(selected_filenames) #convert list to set for faster lookup

    files_to_read = [f for f in all_nist_filenames if f in selected_filenames]
    files_to_read = remove_damaged_files(files_to_read)
    return files_to_read

def process_all_files(): #all files in ../0
    all_nist_filenames = [f for f in os.listdir(nist_jdx_path) if f.endswith('.jdx')]
    files_to_read = remove_damaged_files(all_nist_filenames)
    return files_to_read

def remove_damaged_files(files_to_read):

    non_defective_files = []
    for filename in files_to_read:
        file_path = os.path.join(nist_jdx_path, filename)

        try: 
            with open(file_path, 'r') as file:
                lines = file.readlines()
        except UnicodeDecodeError:
            print(f"Skipping file due to UnicodeDecodeError: {filename}")
            continue

        defective = True
        for line in lines:
            line = line.strip()
            if re.match(r'^[\d\s.]+$', line):
                defective = False
                break
        if not defective:
            non_defective_files.append(filename)

    return non_defective_files

def create_cas_to_func_groups_dict(casnos_txt, compounds_txt):
    cas_to_func_groups = {}

    # Read CAS numbers
    with open(casnos_txt, 'r') as f:
        casnos = [line.strip() for line in f]
    
    # Read compound names
    with open(compounds_txt, 'r') as f:
        compounds = [line.strip() for line in f]

    # Create the dictionary
    count = 1
    for cas, compound in zip(casnos, compounds):
        func_groups = get_func_groups_from_compound_name(compound)
        cas_to_func_groups[cas] = func_groups
        print(f'{count}: {cas}, {func_groups}')
        count += 1
    return cas_to_func_groups

def write_csv(x, y, csv_filename, csv_file_path, cas_to_func_groups):

    filename = csv_filename.split('.')[0]
    compound_func_groups = cas_to_func_groups.get(filename)
              
    with open(csv_file_path, mode='w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(['Wavelength (cm-1)', 'Transmittance', 'Functional Groups'])
        
        if compound_func_groups == 'Compound name not found in pubchem':
            writer.writerows(zip(x, y))            
        else:
            writer.writerows(zip_longest(x, y, compound_func_groups, fillvalue=''))
    
def interpolate_files(files_to_read, cas_to_func_groups):

    for filename in files_to_read:
        #print(filename)
        file_path = os.path.join(nist_jdx_path, filename)

        interpolated_filename = filename.replace('.jdx', '.csv')
        interpolated_path = os.path.join(interpolated_dir, interpolated_filename)

        if interpolate_data(file_path, interpolated_path) is None:
            continue
        
        wavelengths, transmittances = interpolate_data(file_path, interpolated_path)
        
        write_csv(wavelengths, transmittances, interpolated_filename, interpolated_path, cas_to_func_groups)


def IR_data():
    files_to_read = process_all_files()
    cas_funcgroup_dict = create_cas_to_func_groups_dict(casnos_txt, compounds_txt)

    # Convert interpolated_files to a set for faster lookup
    interpolated_files = set(f for f in os.listdir(interpolated_dir) if f.endswith('.csv'))
    # Filter files_to_read to include only those that are not yet interpolated
    files_to_process = [f for f in files_to_read if f.replace('.jdx', '.csv') not in interpolated_files]
    
    files_to_read = [f for f in files_to_read if f in interpolated_files]
    
    interpolate_files(files_to_process, cas_funcgroup_dict)

    wavelengths = np.linspace(4000, 400, 600)
    all_transmittances = []
    all_func_groups = []

    interpolated_files = set(f for f in os.listdir(interpolated_dir))
    for filename in interpolated_files:

        file_path = os.path.join(interpolated_dir, filename)
        df = pd.read_csv(file_path) #read the csv file into a dataframe
        transmittances = df['Transmittance']
        transmittances = transmittances.to_numpy()
        if transmittances.size == 0:
            print(filename)
            continue
        all_transmittances.append(transmittances)
        
        func_groups = df['Functional Groups'].dropna()
        func_groups = func_groups.to_numpy()
        all_func_groups.append(func_groups)

    X = np.vstack(all_transmittances)
    
    #process functional groups for binary encoding
    all_unique_groups = set()
    for func_groups in all_func_groups:
        for func_group in func_groups:
            all_unique_groups.add(func_group)

    with open('all_unique_groups.txt', 'w') as file:
        for func_group in all_unique_groups:
            file.write(func_group + '\n')
  
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

    return X, y


            

