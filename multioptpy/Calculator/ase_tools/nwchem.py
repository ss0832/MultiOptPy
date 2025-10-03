import os
import re


class ASE_NWCHEM: 
    def __init__(self, **kwargs):
        self.atom_obj = kwargs.get('atom_obj', None)
        self.electric_charge_and_multiplicity = kwargs.get('electric_charge_and_multiplicity', None)
        self.input_file = kwargs.get('input_file', None)
        self.functional = kwargs.get('functional', None)
        self.basis_set = kwargs.get('basis_set', None)
        self.memory = kwargs.get('memory', None)

    def run(self):
        from ase.calculators.nwchem import NWChem
        input_dir = os.path.dirname(self.input_file)
        pattern = r"(\d+)([A-Za-z]+)"
        match = re.match(pattern, self.memory.lower())
        if match:
            number = match.group(1)
            unit = match.group(2)
        else:
            raise ValueError("Invalid memory string format")

        calc = NWChem(label=input_dir,
                    xc=self.functional,
                    charge=self.electric_charge_and_multiplicity[0],
                    basis=self.basis_set,
                    memory=number+" "+unit)
        self.atom_obj.set_calculator(calc)
        return self.atom_obj