from cell_edge_vertex import *
from mesh import *
from additional_functions import *



rho = 1
theta_0 = 90
### plus de 80 ça fait des trucs bizarre psk les cellules extérieures ne peuvent pas se detacher
## il faudrait les T1 pour régler ça

##### j'ai changé les keys des edges
#####  avant c'etait le couple de v_id, maintenant c'est un int
######## à vérif dans le code

##### pour plot les edges_id, obligatoirement avec les cells, sinon bug

### mesh hexagonal trop régulier pour les T1 (sauf paramètres extrèmes)
##### faut mesh de base moins régulier

B = 2*rho*np.tan(theta_0/2)

cut = False                 # whether the mesh is cut
grad_S = False              # whether to apply a gradient of incompatibility
grad_L0 = False            # whether to apply a gradient of growth
grad_mode = "center"       # type of gradient to use: "center" or "boundary"
mode = "voronoi"           # use hexagonal cells
num_cells = 100          # number of cells 

S = 1                   # target incompatibility value
L0 = 1                  # target edge length for the hexagons
T1_thr = 1e-1
## à la fin de update_vertices_EE
### check_T1: pb quand y'a une T1, les keys de mon dictionnaire changent 
###### sol 1: break la boucle et rescan tout le dictionnaire
####### sol 1.1: il faudrait allonger le edge qui vient d'avoir un T1 pour éviter qui fasse des T1 à l'infini
####### sol 1.2: on garde en mémoire tous les edges qui ont eu une T1 au fur et à mesure pour les ignorer quand on rescan ()
###### sol 2: on scan tout, fait une liste des edges qui ont besoin de T1 et apres on fait les T1
####### -> il faudrait revérifier la liste avant chaque appel de fonction au cas où une T1 ait modifié la longueur d'un edge ou carrement supprimé l'edge
######################## pour le moment je commente


####### pour voronoi, les coupes ne fonctionnent pas
######## est ce que on pourrait couper en ligne droite quitte à créer de nouvelles cellules? (et tuer celles potentiellement trop petites)

### apres génération voronoi, si on minimise vc S0=1 et L0=1, le tissu rétrécit
#### changer les A0 P0 cible ou changer la taille du cercle pointillé (?)


######### la minimisation est vraiment interminable



# create and initialize the mesh
mesh = Mesh(num_cells=num_cells, S=S, L0=L0, mode=mode, cut=cut, grad_S=grad_S, grad_L0=grad_L0, grad_mode=grad_mode, T1_thr=T1_thr)

mesh.plot(cells=False, edges=True)
mesh.plot(cells=True, edges=False, cells_id=True)
mesh.plot(cells=True, edges=True, edges_id=True)
mesh.plot()

#mesh.equilibrium()

#mesh.plot(cells=False, edges=True)
#mesh.plot(cells=True, edges=False, cells_id=True)
#mesh.plot(cells=True, edges=True, edges_id=True)


"""
mesh.T1_transition(edge_id=1)
#mesh.plot(edges_id=True, cells=True, edges=False)
mesh.T1_transition(edge_id=32)
mesh.T1_transition(edge_id=72)
mesh.T1_transition(edge_id=65)
mesh.T1_transition(edge_id=58)
mesh.plot(edges_id=True, cells=True, edges=True,)
mesh.T1_transition(edge_id=79)
mesh.T1_transition(edge_id=91)
mesh.T1_transition(edge_id=100)
mesh.plot(edges_id=True, cells=True, edges=True,)

mesh.T1_transition(edge_id=150)
mesh.plot(edges_id=True, cells=True, edges=True,)


mesh.T1_transition(edge_id=61)
mesh.plot(cells=False, edges=True)
mesh.plot(edges_id=True, cells=True, cells_id=True,)

mesh.equilibrium()
mesh.plot(cells_id=True, edges_id=True)"""
#mesh.T1_transition(edge_id=8)


# find the equilibrium
# uses the mixed approach combining minimization and explicit Euler 
#mesh.equilibrium()

# plot the mesh
#mesh.plot(cells_id=True, edges_id=True)


#mesh.equilibrium()

#mesh.plot()
