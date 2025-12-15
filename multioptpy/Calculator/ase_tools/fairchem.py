

class ASE_FAIRCHEM:
    def __init__(self, **kwargs):
        self.atom_obj = kwargs.get('atom_obj', None)
        self.electric_charge_and_multiplicity = kwargs.get('electric_charge_and_multiplicity', None)
        self.software_path = kwargs.get('software_path', None)
        self.task_name = kwargs.get('task_name', "omol")
        self.device_mode = kwargs.get('device_mode', "cpu")
        self.software_type = kwargs.get('software_type', None)
        print(f"ASE_FAIRCHEM: software_type = {self.software_type}")
        try:
            from fairchem.core import FAIRChemCalculator
            self.FAIRChemCalculator = FAIRChemCalculator
            from fairchem.core.units.mlip_unit import load_predict_unit
            self.load_predict_unit = load_predict_unit
        except ImportError:
            raise ImportError("FAIRChem.core modules not found")
        
    
    def run(self): # fairchem.core: version 2.x.x
        
        # Load the prediction unit
        predict_unit = self.load_predict_unit(path=self.software_path, device=self.device_mode)
        print(f"ASE_FAIRCHEM: device_mode = {self.device_mode}")
        print(f"ASE_FAIRCHEM: task_name = {self.task_name}")
        # Set up the FAIRChem calculator
        fairchem_calc = self.FAIRChemCalculator(predict_unit=predict_unit, task_name=self.task_name)
        self.atom_obj.info = {"charge": int(self.electric_charge_and_multiplicity[0]),
                              "spin": int(self.electric_charge_and_multiplicity[1])}
        self.atom_obj.calc = fairchem_calc
        
        return self.atom_obj
