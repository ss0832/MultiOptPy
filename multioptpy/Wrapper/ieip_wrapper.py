

import multioptpy.interface
import multioptpy.ieip


class IEIPJob:
    """
    Wrapper class to define and run an iEIP (reaction path search) job
    from a Python script. This replaces ieipmain.py.
    """
    
    def __init__(self, input_folder):
        """
        Initializes the iEIP job settings.
        
        Args:
            input_folder (str): The input folder path.
        """
        
        if not isinstance(input_folder, str):
            raise TypeError("input_folder must be a string.")

        self.input_args = [input_folder]

        # Get default args using the modified ieipparser
        parser = multioptpy.interface.init_parser()
        self.args = multioptpy.interface.ieipparser(parser, self.input_args)
        
        self._ieip_instance = None

    def set_option(self, key, value):
        """Sets a single iEIP job option."""
        if not hasattr(self.args, key):
            print(f"Warning: Option '{key}' is not a default argparse argument.")
        setattr(self.args, key, value)
        print(f"Set option: {key} = {value}")

    def set_options(self, **kwargs):
        """Sets multiple iEIP job options using keyword arguments."""
        print("Setting multiple job options...")
        for key, value in kwargs.items():
            self.set_option(key, value)

    def run(self):
        """Executes the iEIP job."""
        print("\n--- Starting MultiOptPy iEIP Job ---")
        print("Final Settings (args):")
        for key, value in vars(self.args).items():
            print(f"  {key}: {value}")
        print("--------------------------------------")
        
        try:
            self._ieip_instance = multioptpy.ieip.iEIP(self.args)
            self._ieip_instance.run()
            print("--- MultiOptPy Job Finished Successfully ---")
        except Exception as e:
            print(f"--- MultiOptPy Job FAILED ---")
            print(f"An error occurred: {e}")
            raise

    def get_results(self):
        """Retrieves the iEIP instance after the job has been run."""
        if not self._ieip_instance:
            print("Error: .run() must be called before get_results().")
            return None
        return self._ieip_instance

# --- Example Usage ---
if __name__ == "__main__":
    print("=== Example: Defining an iEIP Job ===")
    
    # 1. Define the job
    ieip_job = IEIPJob(input_folder="path/to/my_ieip_input")
    
    # 2. Configure the job
    ieip_job.set_options(
        usextb="GFN2-xTB",
        NSTEP=50,
        opt_method=["FIRE"]
    )
    
    print("\n--- Job Configuration Complete ---")
    print(f"Input folder: {ieip_job.args.INPUT}")
    print(f"Method: {ieip_job.args.usextb}")

    # 3. To run, uncomment:
    # try:
    #     ieip_job.run()
    # except FileNotFoundError:
    #     print("\nError: Input folder not found. Job skipped.")
    # except Exception as e:
    #     print(f"\nAn error occurred during job execution: {e}")