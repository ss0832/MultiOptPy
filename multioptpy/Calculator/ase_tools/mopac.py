import os

class ASE_MOPAC:
    def __init__(self, **kwargs):
        self.atom_obj = kwargs.get('atom_obj', None)
        self.electric_charge_and_multiplicity = kwargs.get('electric_charge_and_multiplicity', None)
        self.input_file = kwargs.get('input_file', None)
        

    def run(self):

        from ase.calculators.mopac import MOPAC
        input_dir = os.path.dirname(self.input_file)
        self.atom_obj.calc = MOPAC(label=input_dir,
                            task="1SCF GRADIENTS DISP",
                            charge=int(self.electric_charge_and_multiplicity[0]),
                            mult=int(self.electric_charge_and_multiplicity[1]))

        return self.atom_obj
