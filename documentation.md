# Project Documentation

This document lists all the classes and functions in the project, along with their parameters and descriptions of what they do.

---

## Classes and Methods

### 1. `Vertex`
Represents a vertex in the system.

#### Methods:

- **`__init__(self, id, position)`**
  - **Parameters:**
    - `id` (int): Unique identifier for the vertex.
    - `position` (list or tuple): Coordinates of the vertex.
  - **Description:** Initializes a vertex with a unique ID and position.

- **`add_cell_id(self, cell_id)`**
  - **Parameters:**
    - `cell_id` (int): The ID of the cell to associate with the vertex.
  - **Description:** Adds a cell ID to the list of cells associated with this vertex, if not already present.

---

### 2. `Edge`
Represents an edge connecting two vertices.

#### Methods:

- **`__init__(self, id, vertex_ids)`**
  - **Parameters:**
    - `id` (int): Unique identifier for the edge.
    - `vertex_ids` (tuple): A tuple of two vertex IDs defining the edge.
  - **Description:** Initializes an edge with a unique ID and two vertex IDs.

- **`add_cell_id(self, cell_id)`**
  - **Parameters:**
    - `cell_id` (int): The ID of the cell to associate with the edge.
  - **Description:** Adds a cell ID to the list of cells sharing this edge, if not already present.

- **`remove_cell_id(self, cell_id)`**
  - **Parameters:**
    - `cell_id` (int): The ID of the cell to remove from the edge's list of associated cells.
  - **Description:** Removes a cell ID from the edge's list of associated cells.

- **`length(self, vertices)`**
  - **Parameters:**
    - `vertices` (dict): A dictionary of vertices indexed by their IDs.
  - **Description:** Calculates and updates the length of the edge based on vertex positions.

---

### 3. `Cell`
Represents a cell in the system.

#### Methods:

- **`__init__(self, id, vertices, num_neigh, relative_position=1, L0=1, alpha=1, beta=1, gamma=0, P0=None, A0=None, S0=2*np.sqrt(np.pi), mode="hexagon")`**
  - **Parameters:**
    - `id` (int): Unique identifier for the cell.
    - `vertices` (list): List of vertices defining the cell.
    - `num_neigh` (int): Number of neighboring cells.
    - `relative_position` (float, optional): Relative position parameter used in growth gradients. Default is 1.
    - `L0` (float, optional): Target length scale for the cell. Default is 1.
    - `alpha`, `beta`, `gamma` (float, optional): Coefficients for area, perimeter, and adhesion terms.
    - `P0`, `A0` (float, optional): Preferred perimeter and area. Default is computed based on mode and L0.
    - `S0` (float, optional): Shape parameter. Default is `2*np.sqrt(np.pi)`.
    - `mode` (str, optional): Preferred geometry of the cell. Options: `"circle"`, `"triangle"`, `"three_triangles"`, `"hexagon"`. Default is `"hexagon"`.
  - **Description:** Initializes a cell with given parameters and computes derived properties.

- **`update_AP(self, vertices)`**
  - **Parameters:**
    - `vertices` (dict): A dictionary of vertices indexed by their IDs.
  - **Description:** Updates the cell's current area and perimeter based on vertex positions.

- **`update_A(self, vertices)`**
  - **Parameters:**
    - `vertices` (dict): A dictionary of vertices indexed by their IDs.
  - **Description:** Updates only the cell's current area.

- **`update_SL(self, new_S, cst, new_L0, mode, grad_S, grad_L0)`**
  - **Parameters:**
    - `new_S` (float): New shape parameter.
    - `cst` (float): Isoperimetric value depending on the cell type.
    - `new_L0` (float): New target size length.
    - `mode` (str): Preferred shape of the cell.
    - `grad_S` (bool): Indicates if perimeter scaling uses a gradient.
    - `grad_L0` (bool): Indicates if length scaling uses a gradient.
  - **Description:** Updates the preferred area and perimeter based on gradients or growth.

- **`anticlockwise(self, vertex_ids, vertices_dict)`**
  - **Parameters:**
    - `vertex_ids` (list): List of vertex IDs defining the cell.
    - `vertices_dict` (dict): Dictionary of vertices indexed by their IDs.
  - **Description:** Orders vertex IDs in anticlockwise order.

- **`area(self, vertices)`**
  - **Parameters:**
    - `vertices` (dict): A dictionary of vertices indexed by their IDs.
  - **Description:** Calculates the 2D polygon area using the shoelace formula.

- **`area_3d(self, vertices)`**
  - **Parameters:**
    - `vertices` (dict): A dictionary of vertices indexed by their IDs.
  - **Description:** Computes the area of a 3D polygon by decomposing it into triangles around the centroid.

- **`perimeter(self, vertices)`**
  - **Parameters:**
    - `vertices` (dict): A dictionary of vertices indexed by their IDs.
  - **Description:** Calculates the perimeter of the polygon.

- **`gradient_area(self, vertex_id, vertices)`**
  - **Parameters:**
    - `vertex_id` (int): ID of the vertex to calculate the gradient for.
    - `vertices` (dict): A dictionary of vertices indexed by their IDs.
  - **Description:** Calculates the gradient of the area with respect to the position of the given vertex.

- **`gradient_perimeter(self, vertex_id, vertices)`**
  - **Parameters:**
    - `vertex_id` (int): ID of the vertex to calculate the gradient for.
    - `vertices` (dict): A dictionary of vertices indexed by their IDs.
  - **Description:** Calculates the gradient of the perimeter with respect to the position of the given vertex.

- **`gradient_adhesion(self, vertex_id, vertices)`**
  - **Parameters:**
    - `vertex_id` (int): ID of the vertex to calculate the gradient for.
    - `vertices` (dict): A dictionary of vertices indexed by their IDs.
  - **Description:** Calculates the gradient of the adhesion energy with respect to the position of the given vertex.

- **`energy_area(self)`**
  - **Description:** Computes the area energy term.

- **`energy_perimeter(self)`**
  - **Description:** Computes the perimeter energy term.

- **`energy_adhesion(self)`**
  - **Description:** Computes the adhesion energy term.

---


### 4. `Mesh`
The `Mesh` class models a mesh system for simulation, offering various operations and utilities to modify, analyze, and visualize the mesh.

#### Methods:


- **`__init__(self, num_cells, dt=0.01, S=1, L0=1, mode="hexagon", alpha=1, beta=1, gamma=0, cut=True, grad_S=False, grad_L0=False, grad_mode="center", theta=90, side_length=1)`**

- **Parameters:**
  - `num_cells` (int): Number of cells.
  - `dt` (float): Time step for the simulation.
  - `S` (float): Initial normalized shape factor.
  - `L0` (float): Target size.
  - `mode` (str): Geometry type of the cells (`"circle"`, `"hexagon"`, `"triangle"`, `"three_triangles"`).
  - `alpha` (float): Model coefficient alpha.
  - `beta` (float): Model coefficient beta.
  - `gamma` (float): Model coefficient gamma.
  - `cut` (bool): Whether to cut the mesh.
  - `grad_S` (bool): Gradient in shape parameter.
  - `grad_L0` (bool): Gradient in size parameter.
  - `grad_mode` (str): Gradient computation mode (`"center"`, `"boundary"`, `"dome"`, `"cone"`).
  - `theta` (float): Target opening angle for `"cone"`.
  - `side_length` (float): Side length of cells.
- **Description:** Initializes the mesh system.

---

##### Mesh Updates

- **`update_vertices_EE(self, scaling_factor=1.0, tolerance=1e-13, dt_max=1, dt_min=1e-3, dt_growth_factor=1.1, threshold=1e-1)`**
  - **Parameters:**
    - `scaling_factor` (float): Factor to scale forces.
    - `tolerance` (float): Threshold below which forces are zeroed out.
    - `dt_max` (float): Maximum time step allowed.
    - `dt_min` (float): Minimum time step allowed.
    - `dt_growth_factor` (float): Growth factor for adaptive time stepping.
    - `threshold` (float): Force magnitude threshold for adjusting time step.
  - **Description:** Updates vertex positions using the explicit Euler method.

- **`update_S0(self, S)`**
  - **Parameters:**
    - `S` (float): New normalized shape parameter.
  - **Description:** Updates the `S0` parameter for all cells.

- **`update_edge_length(self)`**
  - **Parameters:** None
  - **Description:** Computes and updates the length of all edges.

- **`update_cell_AP(self)`**
  - **Parameters:** None
  - **Description:** Computes and updates the area and perimeter for all cells.

- **`update_cell_A(self)`**
  - **Parameters:** None
  - **Description:** Computes and updates the area for all cells.

- **`update_cell_SL(self, new_S, new_L0)`**
  - **Parameters:**
    - `new_S` (float): New normalized shape parameter.
    - `new_L0` (float): New target size parameter.
  - **Description:** Updates the `S` and `L0` parameters for all cells and adjusts `A0` and `P0`.

---


##### Energy and Equilibrium

- **`compute_energy(self)`**
  - **Parameters:** None
  - **Description:** Computes the total and mean energy of the system and stores it.

- **`find_equilibrium(self, tolerance=1e-13, method="trust-constr")`**
  - **Parameters:**
    - `tolerance` (float): Convergence threshold for optimization.
    - `method` (str): Optimization algorithm to use.
  - **Description:** Finds equilibrium positions of vertices by minimizing energy.

---


##### Mesh Generation

- **`generate(self, side_length=1, A0=None, P0=None)`**
  - **Parameters:**
    - `side_length` (float): Side length of cells.
    - `A0` (float or None): Optional target area for cells.
    - `P0` (float or None): Optional target perimeter for cells.
  - **Description:** Generates the mesh based on the specified mode. Defaults to `"hexagon"` if mode is invalid.

- **`generate_hexagons(self, side_length=1, A0=None, P0=None)`**
  - **Parameters:**
    - `side_length` (float): Length of each hexagon side.
    - `A0` (float or None): Optional target area.
    - `P0` (float or None): Optional target perimeter.
  - **Description:** Generates a hexagonal mesh and updates internal attributes.

- **`generate_triangles(self, side_length=1)`**
  - **Parameters:**
    - `side_length` (float): Edge length of triangles.
  - **Description:** Generates a triangular mesh. If `cut` is True, generates only the right half.

- **`generate_half_hybrid_circular_mesh(self, side_length=1, perturbation_scale=0.5)`**
  - **Parameters:**
    - `side_length` (float): Edge length.
    - `perturbation_scale` (float): Scale factor for random displacement of interior vertices.
  - **Description:** Generates a triangular mesh restricted to the right half of a circular domain.

- **`generate_hybrid_circular_mesh(self, side_length=0.1, perturbation_scale=0.5)`**
  - **Parameters:**
    - `side_length` (float): Edge length.
    - `perturbation_scale` (float): Scale factor for random displacement.
  - **Description:** Generates a full hybrid circular mesh with optional perturbations.

- **`three_triangles(self, side_length=1)`**
  - **Parameters:**
    - `side_length` (float): Length of triangle sides.
  - **Description:** Creates a triple triangle mesh.

---


##### Visualization

- **`plot(self, boundary_circle=True, cells=True, cells_id=False, edges=False, vertices=False, right_vertices=False, boundary_points=False, fixed_vertices=False)`**
  - **Parameters:**
    - `boundary_circle` (bool): Display the boundary circle.
    - `cells` (bool): Display cell polygons.
    - `cells_id` (bool): Display cell IDs at centroids.
    - `edges` (bool): Display edges of the cells.
    - `vertices` (bool): Display all vertices.
    - `right_vertices` (bool): Display vertices on right side of cut.
    - `boundary_points` (bool): Display boundary vertices.
    - `fixed_vertices` (bool): Display fixed vertices.
  - **Description:** Plots the mesh with various display options.

- **`plot_3d_sphere(self, rho, h, vertices=False)`**
  - **Parameters:**
    - `rho` (float): Radius of the sphere.
    - `h` (float): Height of the spherical cap.
    - `vertices` (bool): Display vertices.
  - **Description:** Visualizes a spherical cap on a sphere.

- **`plot_3d_cone(self, gb, h, vertices=False)`**
  - **Parameters:**
    - `gb` (float): Base radius of the cone.
    - `h` (float): Height of the cone.
    - `vertices` (bool): Display vertices.
  - **Description:** Visualizes the cone.

- **`plot_mean_energy_force(self, folder_name="test", save=True)`**
  - **Parameters:**
    - `folder_name` (str): Folder where the plot is saved.
    - `save` (bool): Whether to save the plot to disk.
  - **Description:** Plots the evolution of mean energy and mean force over iterations.

---


##### Saving and Exporting

- **`save_3d_dome(self, rho, h, theta, folder_name="test")`**
  - **Parameters:**
    - `rho` (float): Radius of the sphere.
    - `h` (float): Height of the spherical cap.
    - `theta` (float): Central angle.
    - `folder_name` (str): Folder to save the plot.
  - **Description:** Saves a 3D visualization of a dome.

- **`save_3d_cone(self, gb, h, folder_name="test")`**
  - **Parameters:**
    - `gb` (float): Base radius of the cone.
    - `h` (float): Height of the cone.
    - `folder_name` (str): Folder to save the plot.
  - **Description:** Saves a 3D visualization of a cone.

- **`save_flat_dome(self, theta, folder_name)`**
  - **Parameters:**
    - `theta` (float): Central angle.
    - `folder_name` (str): Folder to save the plot.
  - **Description:** Saves the flattened spherical cap.

- **`save_half_simulation(self, num_iterations=None, folder_name="test", boundary_circle=True)`**
  - **Parameters:**
    - `num_iterations` (int or None): Iteration count for labeling.
    - `folder_name` (str): Folder to save the plot.
    - `boundary_circle` (bool): Include boundary circle in the plot.
  - **Description:** Saves the current simulation state with mirrored cells.

---


##### Miscellaneous

- **`random_noise(self, displacement=0.2)`**
  - **Parameters:**
    - `displacement` (float): Maximum random displacement applied to vertices.
  - **Description:** Adds random noise to the positions of interior vertices, ensuring they remain inside the boundary.

- **`rotate_vertices(self, angle_degrees=90)`**
  - **Parameters:**
    - `angle_degrees` (float): Angle to rotate vertices, in degrees (positive = counterclockwise).
  - **Description:** Rotates all vertices around the origin by the specified angle.

- **`copy_scale_mesh(self, scale_factor)`**
  - **Parameters:**
    - `scale_factor` (float): Factor to scale vertex positions and cell properties.
  - **Returns:** A new scaled mesh instance.
  - **Description:** Creates a scaled copy of the mesh.

- **`kill_cells(self, cell_ids)`**
  - **Parameters:**
    - `cell_ids` (int or list of ints): IDs of cells to be removed.
  - **Description:** Removes specified cells and updates associated vertices and edges.

---



##### Boundary Identification

- **`get_boundary_vertex_ids(self)`**
  - **Parameters:** None
  - **Returns:** List of unique boundary vertex IDs.
  - **Description:** Identifies all boundary vertices and marks boundary cells.

- **`get_boundary_vertex_ids_right_of_cut(self, tol=1e-6)`**
  - **Parameters:**
    - `tol` (float): Tolerance for considering vertex on the outer circle.
  - **Description:** Identifies boundary vertices on the right side of the cut and stores their IDs.

---



##### From 3d to 2d embeddings

- **`get_AP_from_other_mesh(self, mesh)`**
  - **Parameters:**
    - `mesh` (Mesh): Another Mesh instance to copy areas and perimeters from.
  - **Description:** Copies target areas (A0) and perimeters (P0) from another mesh.

- **`get_AM1_from_other_mesh(self, mesh)`**
  - **Parameters:**
    - `mesh` (Mesh): Reference mesh instance.
  - **Description:** Updates target perimeter (P0) based on incompatibility from another mesh.

- **`get_AM2_from_other_mesh(self, mesh)`**
  - **Parameters:**
    - `mesh` (Mesh): Reference mesh instance.
  - **Description:** Updates target area (A0) and perimeter (P0) with incompatibility data.

---


##### Shape and Geometry

- **`get_3d_AP(self, h, L)`**
  - **Parameters:**
    - `h` (float): Height of the tetrahedron.
    - `L` (float): Side length.
  - **Returns:** Tuple `(A0, P0)` representing target area and perimeter.
  - **Description:** Computes target area and perimeter assuming a tetrahedral geometry.

- **`compute_opening_angle(self, h, L)`**
  - **Parameters:**
    - `h` (float): Height of the tetrahedron.
    - `L` (float): Side length.
  - **Returns:** Opening angle in degrees.
  - **Description:** Calculates the opening angle for a tetrahedron.



### 5. Standalone Functions

---

- **`S0_with_dome_hexagon(num_cells=2000, folder_name="S0_dome_hexagon", side_length=1)`**
  - **Parameters:**
    - `num_cells` (int, optional): Number of cells to generate in the mesh. Default is 2000.
    - `folder_name` (str, optional): Name of the folder where results will be saved. Default is `"S0_dome_hexagon"`.
    - `side_length` (float, optional): Side length for hexagonal cells. Default is 1.
  - **Description:**  
    Generates a hemispherical hexagonal mesh and computes shape parameters \( S \) and \( S_0 \). It calculates their relationships with angular displacement (theta), saves data and plots showing these relationships.

---

- **`gradual_fit_dome(rho_cst_list=1, num_cells=500, folder_name="Spherical_cap", cut=True, side_length=1, mode="triangle")`**
  - **Parameters:**
    - `rho_cst_list` (float, int, or list, optional): Controls the size of the spherical cap; can be a single value or list of values. Default is 1.
    - `num_cells` (int, optional): Number of cells to generate. Default is 500.
    - `folder_name` (str, optional): Folder name where results and plots will be saved. Default is `"Spherical_cap"`.
    - `cut` (bool, optional): Whether the mesh should be cut. Default is `True`.
    - `side_length` (float, optional): Side length for cells. Default is 1.
    - `mode` (str, optional): Geometry of cells, options include `"triangle"`, `"hexagon"`, or `"circle"`. Default is `"triangle"`.
  - **Description:**  
    Performs flattening of a spherical cap by gradually increasing its size according to a list of rho values. It fits curves to vertex positions on the mesh and saves fitting parameters, data, and plots to files.

---

- **`do_cone(num_cells=500, cut=True, theta=90, mode="triangle", folder_name="Cone")`**
  - **Parameters:**
    - `num_cells` (int, optional): Number of cells to generate in the mesh. Default is 500.
    - `cut` (bool, optional): Whether the mesh should be cut. Default is `True`.
    - `theta` (float, optional): Target opening angle of the cone in degrees. Default is 90.
    - `mode` (str, optional): Mesh geometry type; options include `"triangle"`, `"hexagon"`, or `"circle"`. Default is `"triangle"`.
    - `folder_name` (str, optional): Folder name for saving output files. Default is `"Cone"`.
  - **Description:**  
    Simulates the flattening of a conical mesh with a specified opening angle. It adjusts vertices, performs equilibrium calculations, and saves the flattened mesh and related data.

---
