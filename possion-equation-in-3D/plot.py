import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict


data = np.loadtxt("Output", skiprows=1 , usecols=(0,1) )
r = data[:, 0]
phi = data[:, 1]

avg_phi = defaultdict(list)
for ri, pi in zip(r, phi):
    avg_phi[round(ri, 4)].append(pi)

r_unique = sorted(avg_phi.keys())
phi_avg = [np.mean(avg_phi[ri]) for ri in r_unique]


plt.plot(r_unique, phi_avg, label='Numerical')
plt.plot(r_unique, [1/r if r != 0 else 0 for r in r_unique], 'r--', label='1/r (Coulomb)')

plt.xlabel("Distance r")
plt.ylabel("Potential")
plt.xscale('log')
plt.yscale('log')
plt.grid(True, which="both", ls="--")
plt.legend()
plt.title("Poisson solution vs. 1/r (L=16)")
plt.savefig("potential_L16.png")

