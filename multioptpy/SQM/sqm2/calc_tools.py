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

PRECOMP_FACTORIAL2 = {-1: 1} # (-1)!! = 1
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

PRECOMP_FACTORIAL = {0: 1, 1: 1} # 0! = 1, 1! = 1
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