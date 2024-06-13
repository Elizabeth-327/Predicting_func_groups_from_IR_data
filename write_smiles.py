import pubchempy as pcp

class WriteSmiles:
    def __init__(self, txt_infile, txt_outfile):
        self._txt_infile = txt_infile
        self._txt_outfile = txt_outfile

    def get_smiles(self):
        
        canonical_smiles = {}

        with open(self._txt_infile, 'r') as infile:

            line_count = 1

            for line in infile:

                compound_name = line.strip()
                compound = pcp.get_compounds(compound_name, 'name')
            
                if len(compound) != 0: #compound_name is found in the pubchem database
                    canonical_smiles_of_compound = compound[0].canonical_smiles
                

                if len(compound) == 0: #compound_name is not found in the pubchem database
                    canonical_smiles_of_compound = 'Compound name not found in pubchem'

                canonical_smiles.update({line_count: canonical_smiles_of_compound})
                line_count += 1

        with open(self._txt_outfile, 'w') as outfile:
            for line_count, smiles in canonical_smiles.items():
                outfile.write(f"{line_count}: {smiles}\n")

        updated_canonical_smiles = []
        for smiles in canonical_smiles.values():
            #if smiles != 'Compound name not found in pubchem':
            updated_canonical_smiles.append(smiles)
        return updated_canonical_smiles

        
def get_smiles(compound_name):
    compound = pcp.get_compounds(compound_name, 'name')

    if len(compound) == 0:
        canonical_smiles_of_compound = 'Compound name not found in pubchem'
    else:
        canonical_smiles_of_compound = compound[0].canonical_smiles

    return canonical_smiles_of_compound
                                    
