"""
Create a graph of the Wilson action across configurations.
Uses standard Wilson action (NOT the replica lattice method).
"""

import numpy as np
import lattice_collection as lc
import action_v_cfg
import matplotlib.pyplot as plt

# User parameters
action = 'W'
Nt, Nx, Ny, Nz = 6, 6, 6, 6
beta = 5.7
Nstart = 0
Nend = 8

collection = lc.fn_lattice_collection(action=action, Nt=Nt, Nx=Nx, Ny=Ny, Nz=Nz, beta=beta,
    start=Nstart, end=Nend)

print(np.shape(collection))

actions = []

for Ncfg in range(Nstart, Nend + 1):
    U = np.array(collection[Ncfg], dtype=np.complex128)
    S = action_v_cfg.calc_S(U)
    actions.append(S)

plt.plot(range(Nstart, Nend+1), actions, marker='o')
plt.xlabel('Configuration index')
plt.ylabel('Wilson action S')
plt.title('Wilson action across configurations')
plt.grid(True)
plt.show()



