

class ASE_GAMESSUS:
    def __init__(self, **kwargs):
        self.atom_obj = kwargs.get('atom_obj', None)
        self.electric_charge_and_multiplicity = kwargs.get('electric_charge_and_multiplicity', None)
        self.gamessus_path = kwargs.get('gamessus_path', None)
        self.functional = kwargs.get('functional', None)
        self.basis_set = kwargs.get('basis_set', None)
        self.memory = kwargs.get('memory', None)

    def run(self):
        from ase.calculators.gamess_us import GAMESSUS
        self.atom_obj.calc = GAMESSUS(userscr=self.gamessus_path,
                                contrl=dict(dfttyp=self.functional),
                                charge=self.electric_charge_and_multiplicity[0],
                                mult=self.electric_charge_and_multiplicity[1],
                                basis=self.basis_set)
        return self.atom_obj