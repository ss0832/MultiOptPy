from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("MultiOptPy")
except PackageNotFoundError:
    __version__ = "unknown"
  
from . import interface
from . import neb
from . import ieip
from . import moleculardynamics
from . import optimization

