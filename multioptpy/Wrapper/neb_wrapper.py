
import multioptpy.interface
import multioptpy.neb

class NEBJob:
    """
    Wrapper class to define and run a Nudged Elastic Band (NEB) job
    from a Python script. This replaces nebmain.py.
    """
    
    def __init__(self, input_files):
        """
        Initializes the NEB job settings.
        
        Args:
            input_files (list): 
                A list of input file paths (e.g., ["reactant.xyz", "product.xyz"] 
                or ["img1.xyz", "img2.xyz", ...]).
        """
        
        if not isinstance(input_files, list) or len(input_files) == 0:
            raise TypeError("input_files must be a non-empty list of strings.")

        self.input_args = input_files

        # Get default args using the modified nebparser
        parser = multioptpy.interface.init_parser()
        self.args = multioptpy.interface.nebparser(parser, self.input_args)
        
        self._neb_instance = None

    def set_option(self, key, value):
        """Sets a single NEB job option."""
        if not hasattr(self.args, key):
            print(f"Warning: Option '{key}' is not a default argparse argument.")
        setattr(self.args, key, value)
        print(f"Set option: {key} = {value}")

    def set_options(self, **kwargs):
        """Sets multiple NEB job options using keyword arguments."""
        print("Setting multiple job options...")
        for key, value in kwargs.items():
            self.set_option(key, value)

    def run(self):
        """Executes the NEB job."""
        print("\n--- Starting MultiOptPy NEB Job ---")
        print("Final Settings (args):")
        for key, value in vars(self.args).items():
            print(f"  {key}: {value}")
        print("-----------------------------------")
        
        try:
            self._neb_instance = multioptpy.neb.NEB(self.args)
            self._neb_instance.run()
            print("--- MultiOptPy Job Finished Successfully ---")
        except Exception as e:
            print(f"--- MultiOptPy Job FAILED ---")
            print(f"An error occurred: {e}")
            raise

    def get_results(self):
        """Retrieves the NEB instance after the job has been run."""
        if not self._neb_instance:
            print("Error: .run() must be called before get_results().")
            return None
        return self._neb_instance

# --- Example Usage ---
if __name__ == "__main__":
    print("=== Example: Defining a NEB Job ===")
    
    # 1. Define the job
    neb_job = NEBJob(input_files=["reactant.xyz", "product.xyz"])
    
    # 2. Configure the job
    neb_job.set_options(
        usextb="GFN2-xTB",
        NSTEP=100,
        partition=8,      # 8 intermediate images
        apply_CI_NEB=5    # Apply CI-NEB after 5 steps
    )
    
    print("\n--- Job Configuration Complete ---")
    print(f"Input files: {neb_job.args.INPUT}")
    print(f"Images: {neb_job.args.partition}")

    # 3. To run, uncomment:
    # try:
    #     neb_job.run()
    # except FileNotFoundError:
    #     print("\nError: Input file(s) not found. Job skipped.")
    # except Exception as e:
    #     print(f"\nAn error occurred during job execution: {e}")