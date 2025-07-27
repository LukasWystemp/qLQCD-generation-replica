"""
Entanglement entropy in SU(3) lattice QCD via the Replica Lattice Method

Author: Lukas Wystemp - University of Manchester - Yr3 Physics Summer Project 2025

For an values of alpha [0 - 1] and ensembles creates a set of replica lattices and
analyses the wilson action evolution across configurations and <S_L+1 - S_L> for each alpha
Specify run_number (folder name). 
Only supports action 'W' (Wilson action) with replica boundary conditions for now.

"""

import numpy as np
import os
import gauge_latticeqcd as glqcd
import lattice_collection as lc
import matplotlib.pyplot as plt
from multiprocessing import Process, Pool
from collections import defaultdict
from scipy import integrate

# User parameters
action = 'W'
Nt, Nx, Ny, Nz = 8, 8, 8, 8
beta = 5.7
Nstart = 0
Nend = 120
Ncfg = Nend - Nstart + 1
Nhits = 10
n = 2
s = 2
u0 = 1.0

run_number = 11

alphas = [0, 0.2, 0.4, 0.6, 0.8, 1]
ensembles = [0, 1]

def create_base():
    base_dir = f"RunReplica{run_number}"
    if not os.path.exists(base_dir):
        os.mkdir(base_dir)
        print(f"Created directory: {base_dir}")
    else:
        raise FileExistsError(f"Directory {base_dir} already exists. Please choose a different run number.")
 
    new_base_dir = f"Replica_{action}_{Nt}x{Nx}x{Ny}x{Nz}_b{int(beta * 100)}"
    new_base_dir = os.path.join(base_dir, new_base_dir)
    return new_base_dir

def run_replica_simulation(new_base_dir, alpha, ensemble):
    import os
    import gauge_latticeqcd as glqcd

    dir_name = new_base_dir + f"_alpha{alpha}_ensemble{ensemble}"
    U = glqcd.ReplicaLattice(Nt, Nx, Ny, Nz, beta, u0, n, s)
    matrices = glqcd.create_su3_set()
    U.markov_chain_sweep_replica(Ncfg, matrices, Nhits, dir_name, alpha)


def main_control_run(new_base_dir):
    # Creates and runs simulation

    processes = []

    for alpha in alphas:
        for ensemble in ensembles:
            p = Process(target=run_replica_simulation,
                        args=(new_base_dir, alpha, ensemble))
            p.start()
            processes.append(p)

    print("All processes started, waiting for completion...")
    # Wait for all to finish
    for p in processes:
        p.join()

    print("All processes completed.")



def process_alpha_ensemble(new_base_dir, cfg_start, cfg_end, alpha, ensemble):
    import lattice_collection as lc # reimport for mutliprocessing safety
    import gauge_latticeqcd as glqcd

    dir_name = f"{new_base_dir}_alpha{alpha}_ensemble{ensemble}"
    collection_1 = lc.fn_replica_lattice_collection(Nt, Nx, Ny, Nz, beta, start=cfg_start, end=cfg_end, path=dir_name)

    action_1 = []
    action_2 = []
    int_actions = []
    
    xcutoff_1 = Nx - s
    xcutoff_2 = Nx - (s + 1)
    for cfg in range(cfg_end - cfg_start):
        U1 = np.array(collection_1[cfg], dtype=np.complex128)
       # U2 = np.array(collection_2[cfg], dtype=np.complex128)

        lattice = glqcd.ReplicaLattice(Nt, Nx, Ny, Nz, beta, u0, n, s, U1=U1)

        S1 = lattice.calc_action_replica(xcutoff_1)
        S2 = lattice.calc_action_replica(xcutoff_2)
        action_1.append(S1)
        action_2.append(S2)

        S_int = lattice.calc_S_int(alpha, xcutoff_1, xcutoff_2)
        int_actions.append(S_int)

    return alpha, ensemble, action_1, action_2, int_actions


def plot_actions(new_base_dir):


    jobs = []

    for alpha in alphas:
        for ensemble in ensembles:
            jobs.append((new_base_dir, Nstart, Nend, alpha, ensemble))

    results = []
    with Pool() as pool:
        results = pool.starmap(process_alpha_ensemble, jobs)


    for alpha, ensemble, actions_1, actions_2, S_int in results:
        idx_range = range(Nstart, Nend)
        color_base = { # alpha, colour
            0.0: 'white',
            0.1: 'yellow',
            0.2: 'lime',
            0.4: 'green',
            0.5: 'pink',
            0.6: 'blue',
            0.7: 'purple',
            0.8: 'red',
            0.9: 'grey',
            1.0: 'black'
        }
        color = color_base[alpha]

        plt.plot(idx_range, actions_1, marker='o', label=fr'$S_L$, $\alpha={alpha}$, e={ensemble}', color=color)
        plt.plot(idx_range, actions_2, marker='x', label=fr'$S_{{L+\Delta L}}$, $\alpha={alpha}$, e={ensemble}', color=color)
        plt.plot(idx_range, S_int, marker='^', label=fr'$S_{{\text{{int}}}}$, $\alpha={alpha}$, e={ensemble}', color=color, linestyle='--')

    plt.xlabel('Configuration index')
    plt.ylabel('Wilson action S')
    plt.title('Wilson action across configurations')
    plt.grid(True)
    plt.legend(loc='upper left', fontsize='xx-small')
    plt.savefig(f'Wilson_action_{Nt}x{Nx}x{Ny}x{Nz}_b{int(beta * 100)}_alpha=0-0.4_e=0-1.png', dpi=300)
    plt.show()


def compute_action_diff(cfg_start, cfg_end, dir, alpha, ensemble):
    import lattice_collection as lc
    import gauge_latticeqcd as glqcd

    dir_name = f"{dir}_alpha{alpha}_ensemble{ensemble}"
    collection_1 = lc.fn_replica_lattice_collection(Nt, Nx, Ny, Nz, beta, start=cfg_start, end=cfg_end, path=dir_name)
    actions_1 = []
    actions_2 = []

    xcutoff_1 = Nx - s
    xcutoff_2 = Nx - (s+ 1)

    cut_area = (Nt // n) * Ny * Nz

    for cfg in range(cfg_end - cfg_start):
        U1 = np.array(collection_1[cfg], dtype=np.complex128)
        #U2 = np.array(collection_2[cfg], dtype=np.complex128)

        lattice1 = glqcd.ReplicaLattice(Nt, Nx, Ny, Nz, beta, u0, n, s, U1=U1)

        SL = lattice1.calc_action_replica(xcutoff_1)
        SLp1 = lattice1.calc_action_replica(xcutoff_2)
        actions_1.append(SL)
        actions_2.append(SLp1)

    S_diffs = (np.array(actions_2) - np.array(actions_1)) / ((Nt / n) * Nx * Ny * Nz) # action density
    return alpha, S_diffs
 
def analyse_action_differences(new_base_dir):

    jobs = [(100, Nend, new_base_dir, alpha, e)
            for alpha in alphas for e in ensembles]
    #print(jobs)

    alpha_to_diffs = defaultdict(list)

    with Pool() as pool:
        results = pool.starmap(compute_action_diff, jobs)

    for alpha, S_diff in results:
        alpha_to_diffs[alpha].append(S_diff)
    #print(alpha_to_diffs)


    S_alpha_diffs = []
    S_alpha_std = []
    for alpha in alphas:
        diffs = alpha_to_diffs[alpha]
        #print(alpha, diffs)
        all_diffs = np.concatenate(diffs)
        mean_diff = np.mean(all_diffs)
        std = np.std(all_diffs)
        print(f"Alpha {alpha}: ⟨S_L+1 - S_L⟩ = {mean_diff} ± {std}")
        S_alpha_diffs.append(mean_diff)
        S_alpha_std.append(std)

    plt.figure(figsize=(10, 6))
    plt.errorbar(alphas, S_alpha_diffs, yerr = S_alpha_std, marker='o')
    plt.ylabel(r'$\langle S_{L+1} - S_{L} \rangle$')
    plt.xlabel(r'$\alpha$')
    plt.title('Average Action Difference vs Alpha')
    plt.grid()
    c, l = c_func(S_alpha_diffs)
    plt.text(0.05, 0.1, f"Run number: {run_number}\n$N_t x N_s^3 = {Nt}x{Nx}^3$\nEnsembles: {ensembles[0]}-{ensembles[-1]}\nC({l:.2f}) fm = {c:.5f}", transform=plt.gca().transAxes, fontsize=10, bbox = dict(boxstyle='round', fc='blanchedalmond', ec='orange', alpha=0.5))
    plt.savefig(f"action_diff_vs_alpha_W_{run_number}_{Nt}x{Nx}x{Ny}x{Nz}_alpha{alphas[0]}-{alphas[-1]}_e{ensembles[0]}-{ensembles[-1]}.png", dpi=300)
    plt.show()

    return S_alpha_diffs


def integrate_action_diffs(S_alpha_diffs, max_alpha=1):
    cut = np.array([(a, d) for a, d in zip(alphas, S_alpha_diffs) if a <= max_alpha])
    xs  = cut[:,0]
    ys  = cut[:,1]
    return integrate.simpson(y=ys, x=xs)


def sommer_scale():
    # M. Guagnelli et al. [ALPHA Collaboration], Nucl. Phys. B 535, 389 (1998)
    # equation 2.18
    r0 = 0.5 #fm
    tmp = -1.6805 - 1.7139*(beta - 6) + 0.8155*(beta - 6)**2 - 0.6667 *(beta - 6)**3
    tmp = np.exp(tmp)
    a = r0 * tmp
    return a


def c_func(S_alpha_diffs):
    integrated_diff = integrate_action_diffs(S_alpha_diffs)
    print(f"Integrated action difference: {integrated_diff}")
    a = sommer_scale()
    print(f"Sommer scale a = {a:.4f} fm")
    l = (s + 0.5)*a
    print(f"l = {l:.4f} fm")
    V = Ny * Nz
    C = (s**3 / V) * integrated_diff
    return C, l

def main():
    new_base_dir = create_base()
    main_control_run(new_base_dir)
    plot_actions(new_base_dir)
    S_alpha_diffs = analyse_action_differences(new_base_dir)

    c, l = c_func(S_alpha_diffs)
    print(f"C({l:.2f} fm) = {c:.8f}")

if __name__ == "__main__":
    main()

