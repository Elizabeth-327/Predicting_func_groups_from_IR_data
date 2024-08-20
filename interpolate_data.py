import os
import re
import numpy as np
from jcamp import jcamp_readfile
from scipy import interpolate
import csv

def convert_x(x_in, unit_from, unit_to):
    """Written by Guwon Jung. Correction made in first line by me."""
    """Convert between micrometer and wavenumber."""
    if unit_to == 'micrometers' and unit_from == 'MICROMETERS':
        x_out = x_in
        return x_out
    elif unit_to == 'cm-1' and unit_from in ['1/CM', 'cm-1', '1/cm', 'Wavenumbers (cm-1)']:
        x_out = x_in
        return x_out
    elif unit_to == 'micrometers' and unit_from in ['1/CM', 'cm-1', '1/cm', 'Wavenumbers (cm-1)']:
        x_out = np.array([10 ** 4 / i for i in x_in])
        return x_out
    elif unit_to == 'cm-1' and unit_from == 'MICROMETERS':
        x_out = np.array([10 ** 4 / i for i in x_in])
        return x_out

def convert_y(y_in, unit_from, unit_to):
    """Written by Guwon Jung."""
    """Convert between absorbance and trasmittance."""
    if unit_to == 'transmittance' and unit_from in ['% Transmission', 'TRANSMITTANCE', 'Transmittance']:
        y_out = y_in
        return y_out
    elif unit_to == 'absorbance' and unit_from == 'ABSORBANCE':
        y_out = y_in
        return y_out
    elif unit_to == 'transmittance' and unit_from == 'ABSORBANCE':
        y_out = np.array([1 / 10 ** j for j in y_in])
        return y_out
    elif unit_to == 'absorbance' and unit_from in ['% Transmission', 'TRANSMITTANCE', 'Transmittance']:
        y_out = np.array([np.log10(1 / j) for j in y_in])
        return y_out
    else:
        return None

def get_unique(x_in, y_in):
    """Written by Guwon Jung."""
    """Removes duplicates in x and takes smallest y value for each x value."""
    x_out = sorted(list(set(x_in)), reverse=True)
    y_out = []
    for i in x_out:
        y_temp = []
        for ii, j in zip(x_in, y_in):
            if i == ii:
                y_temp.append(j)
        y_out.append(min(y_temp))
    return x_out, y_out


def interpolate_data(input_file_path, csv_file_path):
    """Written by Guwon Jung. Modifications made by me: function returns None if y is None."""
    jcamp_dict = jcamp_readfile(input_file_path)

    if 'xunits' in jcamp_dict:
        xunit = jcamp_dict['xunits']
    else:
        print(input_file_path)
    if 'yunits' in jcamp_dict:
        yunit = jcamp_dict['yunits']
        
    #why rewrite? isn't the below redundant?
    if 'xlabel' in jcamp_dict:
        xunit = jcamp_dict['xlabel']
    if 'ylabel' in jcamp_dict:
        yunit = jcamp_dict['ylabel']

    x = jcamp_dict['x']
    y = jcamp_dict['y']
    x = convert_x(x, xunit, 'cm-1')
    y = convert_y(y, yunit, 'transmittance')
    if y is None:
        return None
    x_min = min(x)
    x_max = max(x)
    y_min = min(y)
    y_max = max(y)

    if y_max > 1:
        y = [1 if j > 1 else j for j in y]
    if y_min < 0:
        y = [0 if j < 0 else j for j in y]
    #orders x and y arrays in ascending order 
    if x[0] > x[1]:
        x = x[::-1]
        y = y[::-1]
    if x_max > 4000:
        idx = next(i for i, xx in enumerate(x) if xx >= 4000) #returns the index i of the first value xx that satisifies xx >= 4000
        x = x[:idx + 1] #for ex, if x was originally [1,2,4,4001,4002], then x will now be [1,2,4,4001]
        y = y[:idx + 1] #corresponding transmittance values
    if x_min < 400:
        idx = next(i for i, xx in enumerate(x) if xx >= 400)
        #why keep 401?
        x = x[idx - 1:] #for ex, if x was originally [200, 401, 612, 817, 4001], then x will now be [401, 612, 817, 4001]
        y = y[idx - 1:]
    if x_max < 4000:
        #add two new values to the end of the 'x' array for smoothing purposes
        x = np.append(x, [x[-1] + 1, 4000]) #but what if x_max = 3999.9?
        y = np.append(y, [y[-1], y[-1]]) #add duplicates
    if x_min > 400:
        #add two new values to the beginning of the 'x' array for smoothing purposes
        x = np.insert(x, 0, x[0] - 1)
        x = np.insert(x, 0, 400)
        y = np.insert(y, 0, y[0]) #duplicate
        y = np.insert(y, 0, y[0]) #duplicate

    spectrum = get_unique(x, y)
    x = np.linspace(4000, 400, 600)
    f = interpolate.interp1d(spectrum[0], spectrum[1], kind='slinear')
    y = f(x)

    return x, y

    
        
