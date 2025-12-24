class ASE_gxTB_Dev:
    """
    Wrapper class to set up and run g-xTB (preliminary version) calculations via ASE.
    $ pip install pygxtb==0.7.0
    
    """
    def __init__(self, **kwargs):
        
        self.atom_obj = kwargs.get('atom_obj', None)
        self.electric_charge_and_multiplicity = kwargs.get('electric_charge_and_multiplicity', None)
        self.software_path = kwargs.get('software_path', None)
        self.software_type = kwargs.get('software_type', None)
        from pygxtb import PygxTB
        self.pygxtb = PygxTB
        
    def set_calculator(self):
        """
        Sets the ASE calculator object based on software_type.
        """
        
        charge = 0  # Default charge
        if self.electric_charge_and_multiplicity is not None:
            try:
                # Get charge from [charge, multiplicity] list
                charge = int(self.electric_charge_and_multiplicity[0])
            except (IndexError, TypeError, ValueError):
                print(f"Warning: Could not parse charge from {self.electric_charge_and_multiplicity}. Defaulting to 0.")
                pass
        
        # Instantiate GFN0 class and pass the charge
        gxtb_calc = self.pygxtb(charge=charge)
        return gxtb_calc
        

    def run(self):
        """
        Attaches the calculator to the atoms object and returns it.
        """
        calc_obj = self.set_calculator()
        self.atom_obj.calc = calc_obj
        return self.atom_obj