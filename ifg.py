"""
This file is a modified version of ifg.py from [Original Project Name].
Original source: https://github.com/rdkit/rdkit/blob/master/Contrib/IFG/ifg.py 
Original author(s): Richard Hall, Guillaume Godin
Original license: BSD 3-Clause License

Modifications made by Elizabeth Miller:
- Modified the main function to read a file containing a list of compound names and extract their corresponding SMILES; extracted functional groups from SMILES and wrote to output file
- Added functions to write functional groups to output file and extract functional groups from SMILES
"""

from collections import namedtuple

from rdkit import Chem

from write_smiles import WriteSmiles 
from write_smiles import get_smiles

def merge(mol, marked, aset):
  """
  Original function from Original source documented above.
  """
  bset = set()
  for idx in aset:
    atom = mol.GetAtomWithIdx(idx)
    for nbr in atom.GetNeighbors():
      jdx = nbr.GetIdx()
      if jdx in marked:
        marked.remove(jdx)
        bset.add(jdx)
  if not bset:
    return
  merge(mol, marked, bset)
  aset.update(bset)


# atoms connected by non-aromatic double or triple bond to any heteroatom
# c=O should not match (see fig1, box 15).  I think using A instead of * should sort that out?
PATT_DOUBLE_TRIPLE = Chem.MolFromSmarts('A=,#[!#6]')
# atoms in non aromatic carbon-carbon double or triple bonds
PATT_CC_DOUBLE_TRIPLE = Chem.MolFromSmarts('C=,#C')
# acetal carbons, i.e. sp3 carbons connected to tow or more oxygens, nitrogens or sulfurs; these O, N or S atoms must have only single bonds
PATT_ACETAL = Chem.MolFromSmarts('[CX4](-[O,N,S])-[O,N,S]')
# all atoms in oxirane, aziridine and thiirane rings
PATT_OXIRANE_ETC = Chem.MolFromSmarts('[O,N,S]1CC1')

PATT_TUPLE = (PATT_DOUBLE_TRIPLE, PATT_CC_DOUBLE_TRIPLE, PATT_ACETAL, PATT_OXIRANE_ETC)


def identify_functional_groups(mol):
  """
  Original function from original source documented above.
  """
  marked = set()
  #mark all heteroatoms in a molecule, including halogens
  for atom in mol.GetAtoms():
    if atom.GetAtomicNum() not in (6, 1):  # would we ever have hydrogen?
      marked.add(atom.GetIdx())

#mark the four specific types of carbon atom
  for patt in PATT_TUPLE:
    for path in mol.GetSubstructMatches(patt):
      for atomindex in path:
        marked.add(atomindex)

#merge all connected marked atoms to a single FG
  groups = []
  while marked:
    grp = set([marked.pop()])
    merge(mol, marked, grp)
    groups.append(grp)


#extract also connected unmarked carbon atoms
  ifg = namedtuple('IFG', ['atomIds', 'atoms', 'type'])
  ifgs = []
  for g in groups:
    uca = set()
    for atomidx in g:
      for n in mol.GetAtomWithIdx(atomidx).GetNeighbors():
        if n.GetAtomicNum() == 6:
          uca.add(n.GetIdx())
    ifgs.append(
      ifg(atomIds=tuple(list(g)), atoms=Chem.MolFragmentToSmiles(mol, g, canonical=True),
          type=Chem.MolFragmentToSmiles(mol, g.union(uca), canonical=True)))
  return ifgs


def write_functional_groups(funcgroups_txt, all_fgs):
  """
  New function added by Elizabeth Miller.
  """

  with open(funcgroups_txt, 'w') as outfile:

    for fgs in all_fgs:
      if fgs == 'Compound name not found in pubchem':
        outfile.write(f'{fgs}\n')

      else:
        fg_types = ''
        for fg in fgs:
          fg_types += f'{fg.type}, '

        fg_types = fg_types[:-2] #removes ', ' from end of string
        outfile.write(f'{fg_types}\n')


def get_func_groups_from_compound_name(compound_name):
  """
  New function added by Elizabeth Miller.
  """
  smiles = get_smiles(compound_name)

  if smiles == 'Compound name not found in pubchem':
    return 'Compound name not found in pubchem'
  else:
    func_groups = []
    m = Chem.MolFromSmiles(smiles)
    fgs = identify_functional_groups(m)
    for fg in fgs:
      func_groups.append(fg.type)
    return func_groups
  
def main():
  """
  Original function from original source documented above.
  
  Modifications:
  - imported write_smiles.py to extract SMILES values from a list of compounds
  - iterated through the list of SMILES and wrote corresponding functional groups to output file
  """
  
  testsmiles = WriteSmiles('compounds.txt', 'smiles.txt')
  smiles_values = testsmiles.get_smiles()

  all_fgs = []
  for ix, smiles in enumerate(smiles_values):

    if smiles == 'Compound name not found in pubchem':
      print('Compound name not found in pubchem')
      all_fgs.append('Compound name not found in pubchem')

    else:
      m = Chem.MolFromSmiles(smiles)
      fgs = identify_functional_groups(m)

      all_fgs.append(fgs)

      print('%2d: %d fgs' % (ix + 1, len(fgs)), fgs)
      print("")
      for fg in fgs:
        print(f'atoms={fg.atoms}, type={fg.type}')
      print("")

  write_functional_groups('funcgroups.txt', all_fgs)

  

if __name__ == "__main__":
  main()
