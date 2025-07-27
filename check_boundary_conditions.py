import numpy as np

def is_a(txyz, xcutoff):
    if txyz[1] >= xcutoff: 
        return True
    return False

def get_part_bounds(Nt, n):
    """Returns a list of (start, end) tuples defining each partition."""
    if Nt % n != 0:
        raise ValueError(f"Nt = {Nt} must be divisible by n = {n}")
    
    part_size = Nt // n
    bounds = [(i * part_size, (i + 1) * part_size - 1) for i in range(n)]
    return bounds

def find_part(t, bounds):
    """Given a coordinate t, find which part it belongs to."""
    for i, (start, end) in enumerate(bounds):
        if start <= t <= end:
            return i, start, end
    raise ValueError(f"Invalid coordinate t = {t}")

def move(txyz, direction, forward: bool, Nt, n, xcutoff):
    """Move forward (True) or backward (False) with periodic BCs within each partition."""
    t, x, y, z = txyz
    
    if not (0 <= t < Nt and 0 <= x < Nx and 0 <= y < Ny and 0 <= z < Nz):
        raise ValueError(f"Invalid coordinates txyz = {txyz}")

    if direction == 0:
        if is_a(txyz, xcutoff):
            t = (t + 1) % Nt if forward else (t - 1 + Nt) % Nt
        else:
            bounds = get_part_bounds(Nt, n)
            part_idx, start, end = find_part(t, bounds)
            if forward:  
                t = start if t == end else t + 1
            else:
                t = end if t == start else t - 1
    elif direction == 1:
        x = (x + 1) % Nx if forward else (x - 1 + Nx) % Nx
    elif direction == 2:
        y = (y + 1) % Ny if forward else (y - 1 + Ny) % Ny
    elif direction == 3:
        z = (z + 1) % Nz if forward else (z - 1 + Nz) % Nz
    else:
        raise ValueError(f"Invalid direction {direction}, must be 0-3.")

    return [t, x, y, z]

# Example usage
if __name__ == "__main__":
    Nt, Nx, Ny, Nz = 6, 6, 6, 6
    n = 2
    l = 2

    txyz = [6, 2, 5, 3]
    xcutoff = Nx - l
    
    print("Partitions in t:", get_part_bounds(Nt, n))

    direction = 0  # Moving in time direction
    fwd = move(txyz, direction, True, Nt, n, xcutoff)
    bwd = move(txyz, direction, False, Nt, n, xcutoff)
    print(f"txyz = {txyz} in direction {direction}: forward -> {fwd}, backward -> {bwd}")
    
    direction = 1
    fwd = move(txyz, 1, True, Nt, n, xcutoff)
    bwd = move(txyz, 1, False, Nt, n, xcutoff)
    print(f"txyz = {txyz} in direction {direction}: forward -> {fwd}, backward -> {bwd}")
