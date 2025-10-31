
import multioptpy.interface
import multioptpy.moleculardynamics

class MDJob:
    """
    Wrapper class to define and run a Molecular Dynamics (MD) job
    from a Python script. This replaces mdmain.py.
    """
    
    def __init__(self, input_file):
        """
        Initializes the MD job settings.
        
        Args:
            input_file (str): The input psi4 file path.
        """
        
        if not isinstance(input_file, str):
            raise TypeError("input_file must be a string.")

        self.input_args = [input_file]

        # Get default args using the modified mdparser
        parser = multioptpy.interface.init_parser()
        self.args = multioptpy.interface.mdparser(parser, self.input_args)
        
        self._md_instance = None

    def set_option(self, key, value):
        """Sets a single MD job option."""
        if not hasattr(self.args, key):
            print(f"Warning: Option '{key}' is not a default argparse argument.")
        setattr(self.args, key, value)
        print(f"Set option: {key} = {value}")

    def set_options(self, **kwargs):
        """Sets multiple MD job options using keyword arguments."""
        print("Setting multiple job options...")
        for key, value in kwargs.items():
            self.set_option(key, value)

    def run(self):
        """Executes the MD job."""
        print("\n--- Starting MultiOptPy MD Job ---")
        print("Final Settings (args):")
        for key, value in vars(self.args).items():
            print(f"  {key}: {value}")
        print("----------------------------------")
        
        try:
            self._md_instance = multioptpy.moleculardynamics.MD(self.args)
            self._md_instance.run()
            print("--- MultiOptPy Job Finished Successfully ---")
        except Exception as e:
            print(f"--- MultiOptPy Job FAILED ---")
            print(f"An error occurred: {e}")
            raise

    def get_results(self):
        """RetrieVes the MD instance after the job has been run."""
        if not self._md_instance:
            print("Error: .run() must be called before get_results().")
            return None
        return self._md_instance

# --- Example Usage ---
if __name__ == "__main__":
    print("=== Example: Defining an MD Job ===")
    
    # 1. Define the job
    md_job = MDJob(input_file="my_md_input.psi4")
    
    # 2. Configure the job
    md_job.set_options(
        usextb="GFN2-xTB",
        NSTEP=1000,      # 1000 fs
        temperature=300.0,
        timestep=0.5     # 0.5 au
    )
    
    print("\n--- Job Configuration Complete ---")
    print(f"Input file: {md_job.args.INPUT}")
    print(f"Timesteps: {md_job.args.NSTEP}")

    # 3. To run, uncomment:
    # try:
    #     md_job.run()
    # except FileNotFoundError:
    #     print("\nError: Input file not found. Job skipped.")
    # except Exception as e:
    #     print(f"\nAn error occurred during job execution: {e}")