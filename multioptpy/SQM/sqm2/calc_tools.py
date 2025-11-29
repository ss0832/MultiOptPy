import numpy as np

# --- Double Factorial Calculation ---
# (Defined outside the class)
def factorial2_loop(n):
    """Calculates double factorial using a loop."""
    if n < -1:
        raise ValueError("n must be a non-negative integer")
    result = 1
    for i in range(n, 1, -2):
        result *= i
    return result

PRECOMP_FACTORIAL2 = {-1: 1} # (-1)!!  = 1
for i in range(0, 21): # Precompute values
    PRECOMP_FACTORIAL2[i] = factorial2_loop(i)

def factorial2(n):
    """Calculates double factorial using precomputed values or numpy."""
    if n < -1:
        raise ValueError("n must be a non-negative integer")
    elif n in PRECOMP_FACTORIAL2:
        return PRECOMP_FACTORIAL2[n]
    # Calculate if not in precomputed table (usually not needed for large n)
    arr = np.arange(n % 2 + 1, n + 1, 2)
    return int(np.prod(arr))

def factorial_loop(n):
    """Calculates factorial using a loop."""
    if n < 0:
        raise ValueError("n must be a non-negative integer")
    result = 1
    for i in range(2, n + 1):
        result *= i
    return result

PRECOMP_FACTORIAL = {0: 1, 1: 1} # 0!  = 1, 1! = 1
for i in range(2, 21): # Precompute values
    PRECOMP_FACTORIAL[i] = factorial_loop(i)    


def factorial(n):
    """Calculates factorial using precomputed values or numpy."""
    if n < 0:
        raise ValueError("n must be a non-negative integer")
    elif n in PRECOMP_FACTORIAL:
        return PRECOMP_FACTORIAL[n]
    # Calculate if not in precomputed table (usually not needed for large n)
    arr = np.arange(2, n + 1)
    return int(np.prod(arr))
    
def dfactorial(n):
    """
    Returns (2n-1)!! for normalization of Gaussian primitives.
    This is used in the slater2gauss normalization formula.
    
    For angular momentum l, we need (2l-1)!! :
        l=0 (s): dfactorial(1) -> (2*0-1)!! = (-1)!! = 1
        l=1 (p): dfactorial(2) -> (2*1-1)!! = 1!! = 1  
        l=2 (d): dfactorial(3) -> (2*2-1)!!  = 3!! = 3
        l=3 (f): dfactorial(4) -> (2*3-1)!! = 5!! = 15
        l=4 (g): dfactorial(5) -> (2*4-1)!! = 7!! = 105
        
    The input n corresponds to (angmn + 1), so:
        n=1: (-1)!! = 1
        n=2: 1!! = 1
        n=3: 3!! = 3
        n=4: 5!! = 15
        n=5: 7!! = 105
        n=6: 9!! = 945
        n=7: 11!! = 10395
        n=8: 13!! = 135135
    
    This matches the Fortran implementation where dftr(lh) = (2*lh-1)!!
    """
    # DFACTORIAL[n] = (2*(n-1) - 1)!! = (2n - 3)!! 
    # But we want (2l-1)!!  where l = n-1, so (2(n-1)-1)!! = (2n-3)!!
    # Actually, looking at usage: dfactorial(angmn + 1)
    # For angmn=0: dfactorial(1) should give (-1)!! = 1
    # For angmn=1: dfactorial(2) should give 1!! = 1
    # For angmn=2: dfactorial(3) should give 3!! = 3
    # For angmn=3: dfactorial(4) should give 5!! = 15
    
    # Precomputed values: DFACTORIAL[n] = (2n-3)!!
    # Index:  0     1     2     3      4       5        6         7
    # Value: (-3)!! (-1)!! 1!!    3!!    5!!     7!!      9!!       11!! 
    #        1      1      1     3      15      105      945       10395
    
    DFACTORIAL = [1.0, 1.0, 1.0, 3.0, 15.0, 105.0, 945.0, 10395.0, 135135.0]
    n_DFACTORIAL = len(DFACTORIAL)
    
    if n < 0 or n >= n_DFACTORIAL:
        raise ValueError(f"Invalid value for dfactorial: n={n}, must be in range [0, {n_DFACTORIAL - 1}]")
        
    return DFACTORIAL[n]