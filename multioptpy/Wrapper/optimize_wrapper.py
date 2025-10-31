
import multioptpy.interface
import multioptpy.optimization

class OptimizationJob:
    """
    A wrapper class to define and run a multioptpy optimization job
    from a Python script. This replaces optmain.py.
    """
    
    def __init__(self, input_file):
        """
        Initializes the optimization job settings.
        (Omitted for brevity...)
        """
        if isinstance(input_file, str):
            self.input_args = [input_file]
        elif isinstance(input_file, list):
            self.input_args = input_file
        else:
            raise TypeError("input_file must be a string or a list of strings.")
        parser = multioptpy.interface.init_parser()
        self.args = multioptpy.interface.optimizeparser(parser, self.input_args)
        self._optimizer = None

    def set_option(self, key, value):
        """
        Sets a single optimization job option.
        (Omitted for brevity...)
        """
        if not hasattr(self.args, key):
            print(f"Warning: Option '{key}' is not a default argparse argument.")
        setattr(self.args, key, value)
        print(f"Set option: {key} = {value}")

    def set_options(self, **kwargs):
        """
        Sets multiple optimization job options using keyword arguments.
        (Omitted for brevity...)
        """
        print("Setting multiple job options...")
        for key, value in kwargs.items():
            self.set_option(key, value)

    def run(self):
        """
        Executes the optimization job.
        (Omitted for brevity...)
        """
        print("\n--- Starting MultiOptPy Optimization Job ---")
        print("Final Settings (args):")
        for key, value in vars(self.args).items():
            print(f"  {key}: {value}")
        print("--------------------------------------------")
        try:
            self._optimizer = multioptpy.optimization.Optimize(self.args)
            self._optimizer.run()
            print("--- MultiOptPy Job Finished Successfully ---")
        except Exception as e:
            print(f"--- MultiOptPy Job FAILED ---")
            print(f"An error occurred: {e}")
            raise

    def get_results(self):
        """
        Returns the internal optimizer instance after the job has run.
        This instance contains the new file path methods.
        """
        if not self._optimizer:
            print("Error: .run() must be called before get_results().")
            return None
        
        # Return the instance itself, so we can call
        # .get_result_file_path() on it
        return self._optimizer

