import os

class ASE_ORCA:
    def __init__(self, **kwargs):
        self.atom_obj = kwargs.get('atom_obj', None)
        self.electric_charge_and_multiplicity = kwargs.get('electric_charge_and_multiplicity', None)
        self.input_file = kwargs.get('input_file', None)
        self.orca_path = kwargs.get('orca_path', None)
        self.functional = kwargs.get('functional', None)
        self.basis_set = kwargs.get('basis_set', None)
    
    def run(self):
       
        from ase.calculators.orca import ORCA
        input_dir = os.path.dirname(self.input_file)
        self.atom_obj.calc = ORCA(label=input_dir,
                            profile=self.orca_path,
                            charge=int(self.electric_charge_and_multiplicity[0]),
                            mult=int(self.electric_charge_and_multiplicity[1]),
                            orcasimpleinput=self.functional+' '+self.basis_set)
                            #orcablocks='%pal nprocs 16 end')
        return self.atom_obj