class ASE_GFN0:
    """
    Wrapper class to set up and run GFN0-xTB calculations via ASE.
    """
    def __init__(self, **kwargs):
        
        self.atom_obj = kwargs.get('atom_obj', None)
        self.electric_charge_and_multiplicity = kwargs.get('electric_charge_and_multiplicity', None)
        self.software_path = kwargs.get('software_path', None)
        self.software_type = kwargs.get('software_type', None)
        
    def set_calculator(self):
        """
        Sets the ASE calculator object based on software_type.
        """
        import pygfn0 # pygfn0==0.0.3 https://github.com/LiuGaoyong/PyGFN0
        charge = 0  # Default charge
        if self.electric_charge_and_multiplicity is not None:
            try:
                # Get charge from [charge, multiplicity] list
                charge = int(self.electric_charge_and_multiplicity[0])
            except (IndexError, TypeError, ValueError):
                print(f"Warning: Could not parse charge from {self.electric_charge_and_multiplicity}. Defaulting to 0.")
                pass
        
        # Instantiate GFN0 class and pass the charge
        gfn0_calc = pygfn0.GFN0(charge=charge)
        return gfn0_calc
        

    def run(self):
        """
        Attaches the calculator to the atoms object and returns it.
        """
        calc_obj = self.set_calculator()
        self.atom_obj.calc = calc_obj
        return self.atom_obj