from cell_edge_vertex import *
from mesh import *
from additional_functions import *


#### voir si les graphs pvt aider à générer differents meshes


# radius of the flatten cap: sqrt(N3sqrt(3)/2pi)s, s side leght, N number of cells

## R la distance du centre de la cellule par rapport à l'origine

def gamma_R(R, rho):
    return (1+ (R**2)/(4*rho**2))**(-1)

rho = 1
theta_0 = 90
### plus de 80 ça fait des trucs bizarre psk les cellules extérieures ne peuvent pas se detacher
## il faudrait les T1 pour régler ça

B = 2*rho*np.tan(theta_0/2)

cut = False                 # whether the mesh is cut
grad_S = False              # whether to apply a gradient of incompatibility
grad_L0 = False            # whether to apply a gradient of growth
grad_mode = "center"       # type of gradient to use: "center" or "boundary"
mode = "hexagon"           # use hexagonal cells
num_cells = 20            # number of cells 

S = 1                    # target incompatibility value
L0 = 1                   # target edge length for the hexagons

# create and initialize the mesh
mesh = Mesh(num_cells=num_cells, S=S, L0=L0, mode=mode, cut=cut, grad_S=grad_S, grad_L0=grad_L0, grad_mode=grad_mode)

## !! relative_position: 1 au milieu, decroit qd on s'éloigne du centre
#print(mesh.cells[100].relative_position)
relative_positions = [cell.relative_position for cell in mesh.cells.values()]
unique_relative_positions = sorted(set(relative_positions))
#print(unique_relative_positions)

R = np.linspace(1e-5, B, 1000)  # pas 0 pour la division pas R**2


# ---- compute gamma_R on those unique positions ----
gamma_values = gamma_R(np.array(unique_relative_positions), rho)

# ---- plot ----
plt.figure()
plt.plot(unique_relative_positions, gamma_values, marker='o')
plt.xlabel('relative_position')
plt.ylabel(r'$\gamma$')
plt.title(r'$\gamma(\text{relative\_position})$')
plt.grid(True)
plt.show()


"""
## modif A0 en fct de Gamma(R), et P0 en fct de A0
for cell in mesh.cells.values():
    cell.A0 *= gamma_R(R = 1-cell.relative_position, rho = rho)
    cell.P0 = cell.S0 * np.sqrt(cell.A0)
"""
## pb en faisant comme ça, pas de différence entre theta_0 à 45 ou 90 degrés
## psk aucune dependance en theta_0 dans gamma
## donc il faudrait map les relative positions au R qui lui dpd de theta_0 avant de changer A0

unique_relative_positions = sorted(set(relative_positions))
unique_relative_positions = sorted(set(relative_positions), reverse=True)  ## inverse it to have the bigger cells inside
N = len(unique_relative_positions)

# R grid
R = np.linspace(1e-5, B, 1000)
R_grid = np.linspace(1e-5, B, 1000)

# Compute corresponding R values for each unique relative position
alpha = np.linspace(0, 1, N)
R_for_relpos = np.interp(alpha, np.linspace(0, 1, len(R_grid)), R_grid)

# Create a dictionary: relative_position -> matched R
relpos_to_R = dict(zip(unique_relative_positions, R_for_relpos))





for cell in mesh.cells.values():
    # get matched R for this cell
    R_matched = relpos_to_R[cell.relative_position]

    # apply gamma_R using the matched R
    cell.A0 *= gamma_R(R=R_matched, rho=rho)
    cell.P0 = cell.S0 * np.sqrt(cell.A0)




## gamma with mapped R
gamma_mapped = [gamma_R(relpos_to_R[r], rho) for r in unique_relative_positions]
R_mapped     = [relpos_to_R[r] for r in unique_relative_positions]

plt.figure()
plt.plot(R_mapped, gamma_mapped, marker='o')
plt.xlabel('Mapped R')
plt.ylabel(r'$\gamma$')
plt.title(r'$\gamma(R)$ with R mapped from relative_position')
plt.grid(True)
plt.show()






# find the equilibrium
# uses the mixed approach combining minimization and explicit Euler 
mesh.equilibrium()


# plot the mesh
mesh.plot()





W_list = []
R_list = []

for relpos in unique_relative_positions:        # sorted or reverse sorted!
    R_matched = relpos_to_R[relpos]

    Ws = []
    for cell in mesh.cells.values():
        if cell.relative_position == relpos:
            W = cell.alpha*(cell.A - cell.A0)**2 + cell.beta*(cell.P - cell.P0)**2
            Ws.append(W)

    W_list.append(np.mean(Ws))
    R_list.append(R_matched)



# ---- Plot W vs R ----
plt.figure()
plt.scatter(R_list, W_list, c='b')
plt.xlabel("Mapped R")
plt.ylabel("Cell energy W")
plt.title("Cell energy W against mapped R")
plt.grid(True)
plt.show()