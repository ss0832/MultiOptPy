

class ASE_FAIRCHEM:
    def __init__(self, **kwargs):
        self.atom_obj = kwargs.get('atom_obj', None)
        self.electric_charge_and_multiplicity = kwargs.get('electric_charge_and_multiplicity', None)
        self.software_path = kwargs.get('software_path', None)
        self.task_name = "omol"
        self.software_type = kwargs.get('software_type', None)
        print(f"ASE_FAIRCHEM: software_type = {self.software_type}")
        
    
    def run(self): # fairchem.core: version 2.x.x
        try:
            from fairchem.core import FAIRChemCalculator
            from fairchem.core.units.mlip_unit import load_predict_unit
        except ImportError:
            raise ImportError("FAIRChem.core modules not found")
        # Load the prediction unit
        predict_unit = load_predict_unit(path=self.software_path, device="cpu")

        # Set up the FAIRChem calculator
        fairchem_calc = FAIRChemCalculator(predict_unit=predict_unit, task_name=self.task_name)
        self.atom_obj.info = {"charge": int(self.electric_charge_and_multiplicity[0]),
                              "spin": int(self.electric_charge_and_multiplicity[1])}
        self.atom_obj.calc = fairchem_calc
        
        return self.atom_obj
