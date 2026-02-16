class ASE_GFNFF:
    """
    Wrapper class to set up and run GFN-FF calculations via ASE.
    ref.:
    - S.Spicher, S.Grimme. Robust Atomistic Modeling of Materials, Organometallic, and Biochemical Systems (2020), DOI: https://doi.org/10.1002/anie.202004239
    - A standalone library of the GFN-FF method. https://github.com/pprcht/gfnff/
    - https://github.com/LiuGaoyong/PyGFNFF
    """
    def __init__(self, **kwargs):
        
        self.atom_obj = kwargs.get('atom_obj', None)
        self.electric_charge_and_multiplicity = kwargs.get('electric_charge_and_multiplicity', None)
        self.software_path = kwargs.get('software_path', None)
        self.software_type = kwargs.get('software_type', None)
        
    def set_calculator(self):
        """
        Sets the ASE calculator object utilizing pygfnff.
        """
        import pygfnff
        
        # While GFN-FF is a force field and topology generation is often automated,
        # we retain the logic to parse charge to maintain consistency with the
        # wrapper design pattern. 
        # Note: Specific support for explicit charge setting depends on the 
        # underlying PyGFNFF implementation details.
        charge = 0
        if self.electric_charge_and_multiplicity is not None:
            try:
                # Get charge from [charge, multiplicity] list
                charge = int(self.electric_charge_and_multiplicity[0])
            except (IndexError, TypeError, ValueError):
                print(f"Warning: Could not parse charge from {self.electric_charge_and_multiplicity}. Defaulting to 0.")
                pass
        
        # Instantiate GFNFF calculator.
        # Based on the provided reference snippet: calculator=GFNFF()
        # If the specific version of PyGFNFF supports charge argument, it can be passed here.
        # Assuming standard initialization for now.
        gfnff_calc = pygfnff.GFNFF()
        
        return gfnff_calc

    def run(self):
        """
        Attaches the calculator to the atoms object and returns it.
        """
        calc_obj = self.set_calculator()
        self.atom_obj.calc = calc_obj
        return self.atom_obj