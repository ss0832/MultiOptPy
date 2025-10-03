


class ASE_MACE:
    def __init__(self, **kwargs):
        
        self.atom_obj = kwargs.get('atom_obj', None)
        self.electric_charge_and_multiplicity = kwargs.get('electric_charge_and_multiplicity', None)
        self.software_path = kwargs.get('software_path', None)
        self.software_type = kwargs.get('software_type', None)
        
    def set_nnp(self):
        if self.software_type == "MACE_MP":
            from mace.calculators import mace_mp
            mace_mp = mace_mp()
            return mace_mp
        elif self.software_type == "MACE_OFF":
            from mace.calculators import mace_off
            mace_off = mace_off()
            return mace_off
        else:
            raise ValueError(f"Unsupported software type: {self.software_type}")


    def run(self):
        nnp_obj = self.set_nnp()
        self.atom_obj.calc = nnp_obj
        return self.atom_obj
