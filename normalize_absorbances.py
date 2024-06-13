import os
import re

def normalize_ir_spectrum(input_file_path, output_file_path):
    with open(input_file_path, 'r') as input_file:
        lines = input_file.readlines()

    metadata = []
    newlines = []
    
    for line in lines:
        line = line.strip()

        #get y factor
        if line.startswith('##YFACTOR'):
            yfactor = line.split('=')[1]
            line = '##YFACTOR=1.0'

        #get metadata
        if not re.match(r'^[\d\s.]+$', line):
            metadata.append(line)

        #get wavelengths and normalize absorbances
        else:
            wavelength = line.split(' ')[0]
            absorbances = line.split(' ')[1:]

            normalized_absorbances = []
            for absorbance in absorbances:
                try:
                    normalized_absorbance = float(absorbance)*float(yfactor)
                except ValueError:
                    print(input_file_path)
                normalized_absorbances.append(str(normalized_absorbance))

            normalized_absorbances = (' ').join(normalized_absorbances) #convert to string
            newline = wavelength + ' ' + normalized_absorbances
            newlines.append(newline)

    endline = metadata.pop() #pops '##END='

    
    with open(output_file_path, 'w') as output_file:

        for line in metadata:
            output_file.write(f'{line}\n')

        for newline in newlines:
            output_file.write(f'{newline}\n')

        output_file.write(endline)

    
def main():

    script_dir = os.path.dirname(__file__) #current directory
    input_file_path = os.path.join(script_dir, '..', '0', '74-84-0.jdx')

    normalize_ir_spectrum(input_file_path, '74-84-0_normalized.jdx')

main()
        
