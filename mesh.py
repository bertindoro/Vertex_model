import numpy as np
from cell_edge_vertex import *
from additional_functions import *
from shapely.geometry import Polygon
import matplotlib.pyplot as plt
import os
from scipy.optimize import minimize
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.spatial import Delaunay
from matplotlib.colors import Normalize
import matplotlib.cm as cm

from scipy.spatial import Voronoi



class Mesh:
    def __init__(self, num_cells, dt=0.01, S = 1, L0 = 1, mode = "hexagon", alpha = 1, beta = 1, gamma = 0, cut=True, grad_S=False, grad_L0=False, grad_mode="center", theta=90, side_length=1, T1_thr = 1e-2):
        """
        Initialize the mesh with parameters
        
        Parameters:
        - num_cells: number of cells 
        - dt: time step for simulation
        - S: initial normalized shape factor
        - L0: target size
        - mode: geometry type of the cell ("circle", "hexagon", "triangle", "three_triangles")
        - alpha, beta, gamma: coefficients of the model
        - cut: whether to cut the mesh
        - grad_S: if there is a gradient in shape parameter
        - grad_L0: if there is a gradient in size parameter
        - grad_mode: mode of gradient computation ("center", "boundary", "dome", "cone")
        - theta: target opening angle for "cone"
        - side_length
        - T1_thr: arbitrary threshold for T1 transitions
        - seed: to have a
        """
        self.num_cells = num_cells
        self.dt = dt
        self.mode = mode
        self.S_end = S
        self.L0_end = L0
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.cut = cut
        self.L0 = L0
        if mode == "circle":
            self.cst_S0 = 2*np.sqrt(np.pi)
        elif mode == "triangle":
            self.cst_S0 = 6/np.sqrt(np.sqrt(3))    # my value for an equilateral trialgle
        elif mode == "three_triangles":
            self.num_cells = 3
            self.cst_S0 = (2+2/np.cos(np.pi/6)) /np.sqrt(np.tan(np.pi/6))
        else: 
            self.cst_S0 = np.sqrt(24/np.sqrt(3))
        self.S0 = S * self.cst_S0
        self.vertices = {}
        self.cells = {}
        self.edges = {}  
        self.radius = None
        self.total_force = [0]
        self.mean_force = [0]
        self.total_energy = []
        self.mean_energy = []
        self.grad_S = grad_S
        self.grad_L0 = grad_L0
        self.grad_mode = grad_mode
        self.theta = theta
        self.T1_thr = T1_thr
        self.right_side_vertex_ids = None
        self.fix_vertex_xy = []
        self.fix_vertex_x = []
        self.fix_vertex_y = []
        self.generate(side_length=side_length)


    def update_vertices_EE(self, scaling_factor=1.0, tolerance=1e-13, dt_max=1, dt_min=1e-3, dt_growth_factor=1.1, threshold = 1e-1):
        """
        Update vertex positions using explicit Euler
        
        Parameters:
        - scaling_factor: factor to scale forces
        - tolerance: threshold to zero out small forces
        - dt_max: maximum time step
        - dt_min: minimum time step
        - dt_growth_factor: growth factor for adaptive time stepping
        - threshold: force magnitude threshold for time step adjustment
        """
        forces = {}

        # Calculate forces for all vertices
        for vertex_id, vertex in self.vertices.items():
            total_force = np.array([0.0, 0.0])
                
            for cell_id in vertex.cell_ids:
                cell = self.cells[cell_id]
                grad_area = cell.gradient_area(vertex.id, self.vertices)
                grad_perimeter = cell.gradient_perimeter(vertex.id, self.vertices)
                grad_adhesion = cell.gradient_adhesion(vertex.id, self.vertices)

                # Zero out small gradient components
                grad_area = np.where(np.abs(grad_area) < tolerance, 0.0, grad_area)
                grad_perimeter = np.where(np.abs(grad_perimeter) < tolerance, 0.0, grad_perimeter)
                grad_adhesion = np.where(np.abs(grad_adhesion) < tolerance, 0.0, grad_adhesion)

                total_force += grad_perimeter + grad_area + grad_adhesion
            if vertex_id in self.fix_vertex_xy:
                total_force[:] = 0
            else:
                if vertex_id in self.fix_vertex_x:
                    total_force[0] = 0
                if vertex_id in self.fix_vertex_y:
                    total_force[1] = 0
            forces[vertex_id] = total_force * scaling_factor  # Apply scaling

        # Adjust dt based on forces
        force_magnitude = np.array([np.linalg.norm(force) for force in forces.values()])
        force_mean = np.mean(force_magnitude) #mean
        
        if force_mean < self.mean_force[-1] and force_mean < threshold:
            self.dt = min(self.dt * dt_growth_factor, dt_max)
        else:
            self.dt = dt_min

          
        # Update vertex positions
        for vertex_id, force in forces.items():
            self.vertices[vertex_id].position -= self.dt * force  # Move vertex based on the force

        # Additional updates
        self.update_cell_AP()
        self.update_edge_length()
        #self.check_T1()
        self.mean_force.append(force_mean)
        self.total_force.append(sum(force_magnitude)) 
        self.compute_energy() 



    def update_S0(self, S):
        """
        Update S0 parameter for all cells
        
        Parameters:
        - S: new normalized shape parameter
        """    
        self.S0 = S * self.cst_S0
        for cell in self.cells.values():
            cell.P0 = self.S0 * np.sqrt(cell.A0)


    def update_edge_length(self):
        """
        Compute and update the length of all edges
        """
        for edge in self.edges.values():
            edge.length(self.vertices)


    def update_cell_AP(self):
        """
        Compute and update the area and perimeter for all cells
        """
        for cell in self.cells.values():
            cell.update_AP(self.vertices)


    def update_cell_A(self):
        """
        Compute and update the area for all cells
        """
        for cell in self.cells.values():
            cell.update_A(self.vertices)


    def update_cell_SL(self, new_S , new_L0):
        """
        Update the S and L0 parameters for all cells and adjust A0 and P0 
        
        Parameters:
        - new_S: new normalized shape parameter
        - new_L0: new target size parameter
        """
        self.S0 = self.cst_S0*new_S
        self.L0 = new_L0
        for cell in self.cells.values():
            cell.update_SL(new_S, self.cst_S0, self.L0, self.mode, self.grad_S, self.grad_L0)
   

    def compute_energy(self):
        """
        Compute the total and mean energy of the system and store it
        """
        total = 0
        for cell in self.cells.values():
            total += cell.energy_area() + cell.energy_perimeter() + cell.energy_adhesion()
        self.total_energy.append(total)
        self.mean_energy.append(total/self.num_cells)


    def find_equilibrium(self, tolerance=1e-13, method='trust-constr'):
        """
        Find the equilibrium positions of the vertices by minimizing energy

        Parameters:
        - tolerance: convergence threshold for optimization
        - method: optimization algorithm to use
        """
        

        def compute_gradients(positions):
            """
            Compute forces for all vertices given their positions
            """
            forces = np.zeros_like(positions)  # Initialize forces as a flat array
            idx = 0

            # Map the flat positions array back to vertices
            for vertex_id, vertex in self.vertices.items():
                vertex.position = positions[idx:idx + 2]
                idx += 2
            self.update_cell_AP()

            # Compute forces for each vertex
            for vertex_id, vertex in self.vertices.items():
                total_force = np.array([0.0, 0.0])

                for cell_id in vertex.cell_ids:
                    cell = self.cells[cell_id]

                    # Compute gradients for the vertex
                    grad_area = cell.gradient_area(vertex.id, self.vertices)
                    grad_perimeter = cell.gradient_perimeter(vertex.id, self.vertices)
                    grad_adhesion = cell.gradient_adhesion(vertex.id, self.vertices)

                    # Zero out small gradient components
                    grad_area = np.where(np.abs(grad_area) < tolerance, 0.0, grad_area)
                    grad_perimeter = np.where(np.abs(grad_perimeter) < tolerance, 0.0, grad_perimeter)
                    grad_adhesion = np.where(np.abs(grad_adhesion) < tolerance, 0.0, grad_adhesion)

                    total_force += grad_area + grad_perimeter + grad_adhesion
                                # Apply fixed‐vertex constraints to force

                if vertex_id in getattr(self, 'fix_vertex_xy', ()):
                    total_force[:] = 0.0
                else:
                    if vertex_id in getattr(self, 'fix_vertex_x', ()):
                        total_force[0] = 0.0
                    if vertex_id in getattr(self, 'fix_vertex_y', ()):
                        total_force[1] = 0.0

                vertex_index = list(self.vertices.keys()).index(vertex_id)
                forces[2 * vertex_index:2 * vertex_index + 2] = total_force

            return forces.flatten()

        def compute_energy(positions):
            """
            Compute the energy of the system based on vertex positions
            """
            idx = 0
            for vertex_id, vertex in self.vertices.items():
                vertex.position = positions[idx:idx + 2]
                idx += 2
            self.update_cell_AP()

            # Accumulate energy contributions
            total_energy = 0.0
            area_energy = 0.0
            perimeter_energy = 0.0
            adhesion_energy = 0.0

            for cell in self.cells.values():
                area_energy += cell.energy_area()
                perimeter_energy += cell.energy_perimeter()
                adhesion_energy += cell.energy_adhesion()

                #useless
                area_energy = np.where(np.abs(area_energy) < tolerance, 0.0, area_energy)
                perimeter_energy = np.where(np.abs(perimeter_energy) < tolerance, 0.0, perimeter_energy)
                adhesion_energy = np.where(np.abs(adhesion_energy) < tolerance, 0.0, adhesion_energy)

            total_energy = area_energy + perimeter_energy + adhesion_energy

            return total_energy



        # Initial positions
        initial_positions = np.array([vertex.position for vertex in self.vertices.values()]).flatten()


        # Optimize using chosen method
        result = minimize(
            fun=compute_energy,
            x0=initial_positions,
            method=method,
            jac=compute_gradients,
            tol=tolerance,
            options={"disp": False}  # Display optimization details
        )

        #print(f"Optimization success: {result.success}, Message: {result.message}")

        # Update vertex positions with optimized results
        optimized_positions = result.x
        idx = 0 
        for vertex_id, vertex in self.vertices.items():
            vertex.position = optimized_positions[idx:idx + 2]
            idx += 2

        # Debugging: Check final energy and gradients
        final_energy = compute_energy(optimized_positions)
        final_gradients = compute_gradients(optimized_positions)
        #print(f"Final energy: {final_energy}")
        #print(f"Final gradients (should be close to zero): {np.mean(final_gradients)}")
        self.update_vertices_EE()  ## to have the mean forces update


    def generate(self, side_length=1, A0=None, P0=None):
        """
        Generate the mesh based on the specified mode. 
        If wrong input, the mode is set to "hexagon" 
        
        Parameters:
        - side_length: side length of cells
        - A0: optional target area for cells
        - P0: optional target perimeter for cells
        """
        if self.mode=="triangles":
            self.mode = "triangle" 
        if self.mode=="hexagons":
            self.mode = "hexagon" 
        if self.mode=="circles":
            self.mode = "circle" 
        if self.mode == "three_triangle":
            self.mode = "three_triangles"
        if self.mode=="three_triangles":
            self.three_triangles(side_length=side_length)
        elif self.mode=="triangle":
            self.generate_triangles(side_length=side_length)
        elif self.mode=="voronoi":
            self.generate_voronoi(side_length=side_length)
        else:
            if self.mode != "circle":
                self.mode = "hexagon"
            self.generate_hexagons(side_length=side_length, A0=A0, P0=P0)
        self.get_boundary_vertex_ids()


    def get_starting_state_EE(self, threshold=5e-4):
        """
        Using EE until the mean forces drop under a given threshold

        Parameter:
        - threshold: threshold value (default: 5e-4)
        """
        self.update_vertices_EE()
        k=1
        while self.mean_force[-1]>threshold:
            self.update_vertices_EE()
            k+=1
            
  


    def equilibrium(self):
        """
        Iteratively finds equilibrium states and adapts thresholds based on the mean force

        This method uses predefined thresholds to refine the state of the system until
        equilibrium is achieved. Thresholds are skipped dynamically depending on the current
        value of `mean_force`
        """
        # Predefined thresholds for get_starting_state_EE()
        #thresholds = [5e-3, 1e-3, 7.5e-4, 5e-4, 2.5e-4, 1e-4, 7.5e-5, 5e-5, 4e-5, 3e-5, 2.5e-5, 2e-5, 1.75e-5, 1.5e-5, 1.25e-5, 1e-5 ]
        thresholds = [5e-3, 1e-3, 7.5e-4, 5e-4, 2.5e-4, 1e-4 , 7.5e-5, 5e-5]


        while thresholds:
            # Use minimization function
            self.find_equilibrium()

            # Check the mean force value
            mean_force = self.mean_force[-1]
            print(f"Current mean force: {mean_force}, target: {thresholds[-1]}")

            # Determine the next threshold
            for i, threshold in enumerate(thresholds):
                if mean_force < threshold:
                    next_threshold = threshold
                    thresholds = thresholds[i + 1:]  # Skip thresholds up to the selected one
                    break
            else:
                # If no threshold is smaller, use the last one
                next_threshold = thresholds[-1]
                thresholds = []  # Clear thresholds to end the loop

            # Update state
            self.get_starting_state_EE(threshold=next_threshold)


    def generate_hexagons(self, side_length=1, A0=None, P0=None):
        """
        Generate a mesh of hexagonal cells based on specified parameters
        Updates internal attributes such as vertices, edges, cells, and cell properties
        
        Parameters:
        - side_length: length of each side of the hexagons (default: 1)
        - A0: optional target area for the hexagonal cells
        - P0: optional target perimeter for the hexagonal cells
        """
        x_length = self.L0_end
        
        if self.grad_mode=="dome" or self.grad_mode=="cone":
            x_length = side_length

        y_length = np.sqrt(x_length**2 - (x_length / 2)**2)

        vertices = {}
        vertices_by_id = {}
        hexagons = {}

        directions = [
            (1.5 * x_length, y_length),
            (1.5 * x_length, -y_length),
            (0, -2 * y_length),
            (-1.5 * x_length, -y_length),
            (-1.5 * x_length, y_length),
            (0, 2 * y_length),
        ]

        def add_vertex(position):
            pos_tuple = tuple(np.round(position, decimals=6))
            if pos_tuple not in vertices:
                vertex_id = len(vertices) + 1
                vertex = Vertex(id=vertex_id, position=position)
                vertices[pos_tuple] = vertex
                vertices_by_id[vertex_id] = vertex
            return vertices[pos_tuple]

        def add_edge_old(vertex_ids):
            edge_key = tuple(sorted(vertex_ids))
            if edge_key not in self.edges:
                edge_id = len(self.edges) + 1
                self.edges[edge_key] = Edge(id=edge_id, vertex_ids=vertex_ids)
            return self.edges[edge_key]
        
        def add_edge(vertex_ids):
            edge_key = tuple(sorted(vertex_ids))

            # si l'arête existe déjà (on la cherche par ses vertices)
            for edge in self.edges.values():
                if tuple(sorted(edge.vertex_ids)) == edge_key:
                    return edge

            # sinon création
            edge_id = len(self.edges) + 1
            edge = Edge(id=edge_id, vertex_ids=vertex_ids)

            # IMPORTANT : clé = edge.id (PAS edge_key)
            self.edges[edge_id] = edge

            return edge

        centers = [(0.0, 0.0)]
        visited = set(tuple(np.round(center, decimals=6)) for center in centers)

        while len(centers) < int(self.num_cells*3):
            new_centers = []
            for center in centers:
                for direction in directions:
                    new_center = (center[0] + direction[0], center[1] + direction[1])
                    rounded_center = tuple(np.round(new_center, decimals=6))
                    if rounded_center not in visited:
                        new_centers.append(new_center)
                        visited.add(rounded_center)
                    if len(centers) + len(new_centers) >= int(self.num_cells*3):
                        break
                if len(centers) + len(new_centers) >= int(self.num_cells*3):
                    break
            centers.extend(new_centers[:int(self.num_cells*3) - len(centers)])

        # Do a circular shape for the outside
        self.radius = 1.1 * np.linalg.norm(np.abs(centers[self.num_cells - 1]))
        centers = [center for center in centers if np.linalg.norm(center) <= self.radius]    

        if self.cut:
            centers = [center for center in centers if np.abs(center[0]) > 0.1 or center[1] > 0.1]

        

        # Precompute neighbors for each center
        neighbor_distance = 2 * y_length  # Distance between centers of neighboring hexagons
        neighbors_count = {}
        for i, center1 in enumerate(centers):
            count = 0
            for j, center2 in enumerate(centers):
                if i != j:
                    distance = np.linalg.norm(np.array(center1) - np.array(center2))
                    if np.isclose(distance, neighbor_distance, atol=1e-6):
                        count += 1
            neighbors_count[tuple(np.round(center1, decimals=6))] = count


        # A center is on the boundary if it has fewer than 6 neighbors
        boundary_centers = set()
        for center in centers:
            if neighbors_count[tuple(np.round(center, decimals=6))] < 6:
                boundary_centers.add(center)

        #Calculate distance to the nearest boundary center
        def distance_to_boundary(center, boundary_centers):
            return min(np.linalg.norm(np.array(center) - np.array(boundary)) for boundary in boundary_centers)

        max_distance_from_boundary = max(distance_to_boundary(center, boundary_centers) for center in centers)

        #Generate hexagons and compute the distance to boundary for each center
        for center in centers:
            cx, cy = center
            hex_vertices = []
            edges = []
            for i in range(6):
                angle = 2 * np.pi * i / 6
                x = cx + x_length * np.cos(angle)
                y = cy + x_length * np.sin(angle)
                vertex = add_vertex((x, y))
                hex_vertices.append(vertex)

            hex_vertex_ids = [v.id for v in hex_vertices]
            hex_edges = [add_edge((hex_vertex_ids[i], hex_vertex_ids[(i + 1) % 6]))
                        for i in range(6)]

            if self.grad_mode == "center" or self.grad_mode=="dome" or self.grad_mode=="cone":
                relative_position = 1 - min(1, round(np.linalg.norm(center)/self.radius, 10))
            else:
                # Calculate the distance to boundary (from the nearest boundary center)
                distance_from_boundary = distance_to_boundary(center, boundary_centers)
                relative_position = 0
                if max_distance_from_boundary != 0:
                    start_growth = 1 #the minimal growth will be (1-start_growth)*L0
                                #### maybe
                    relative_position = round((distance_from_boundary / max_distance_from_boundary)*start_growth, 10) 

            num_neigh = neighbors_count[tuple(np.round(center, decimals=6))]

            hex_cell = Cell(
                id=len(hexagons) + 1,
                vertices=hex_vertices,
                num_neigh=num_neigh,
                relative_position=relative_position,
                L0=side_length,
                alpha=self.alpha,
                beta=self.beta,
                gamma=self.gamma,
                S0=self.cst_S0,
                A0=A0,
                P0=P0,
                mode=self.mode)
            hexagons[len(hexagons) + 1] = hex_cell

            # Associate edges with the cell
            for edge in hex_edges:
                edge.add_cell_id(hex_cell.id)

        # After all cells are created
        self.vertices = vertices_by_id
        self.cells = hexagons
        self.update_cell_AP()
        self.update_edge_length()
        self.num_cells = len(self.cells)
        if self.grad_mode=="dome":
            self.radius = np.sqrt((self.num_cells)*3*np.sqrt(3)/(2*np.pi))*side_length
        elif self.grad_mode=="cone":
            self.radius = side_length*np.sqrt((3*self.num_cells*np.sqrt(3))/(2*np.pi))
        else:
            self.radius = self.radius /x_length *side_length
        self.compute_energy()
        self.update_cell_SL(self.S_end, self.L0_end)
        self.get_boundary_vertex_ids()
                # vertices to track on the cut
        if self.cut:
            tol = side_length/(1.5)
            self.right_side_vertex_ids = []

            for vid, vertex in self.vertices.items():
                x, y = vertex.position
                # on the segment x=0, y in [0,-radius]
                if 0 < x and x < tol and (y <= tol) and (y >= - self.radius - tol):
                    self.right_side_vertex_ids.append(vid)
        self.rotate_vertices()
    

    def plot(self, boundary_circle=True, cells=True, cells_id=False, edges=False, edges_id=False,
         vertices=False, right_vertices=False, boundary_points=False, fixed_vertices=False):
        """
        Plot the mesh with some options. Default display the boundary circle and the cells
        
        Parameters:
        - boundary_circle: whether to display the boundary circle (default: True)
        - cells: whether to display cell polygons (default: True)
        - cells_id: whether to display cell IDs at their centroids (default: False)
        - edges: whether to display the edges of the cells (default: False)
        - vertices: whether to display all vertices (default: False)
        - right_vertices: whether to display vertices on the right side of the cut (default: False)
        - boundary_points: whether to display boundary vertices (default: False)
        - fixed_vertices: whether to display fixed vertices (default: False)
        """
        plt.figure(figsize=(8, 8))

        # Draw the boundary circle
        if boundary_circle:
            circle = plt.Circle((0, 0), self.radius, color='black', linestyle='--', fill=False, linewidth=3)
            plt.gca().add_patch(circle)


        # Cells
        for cell in self.cells.values():
            positions = [self.vertices[v_id].position for v_id in cell.vertex_ids]
            positions.append(positions[0])  # Close the polygon
            polygon = Polygon(positions)

            # Plot the cell
            x, y = polygon.exterior.xy
            if cells:
                plt.plot(x, y, 'b-')
            
        
            # Calculate and plot the cell ID at its centroid
            if cells_id:
                centroid = polygon.centroid
                plt.text(centroid.x, centroid.y, str(cell.id), color='black', fontsize=10,
                        ha='center', va='center')
            
        # Edges
        for edge in self.edges.values():
            v1, v2 = edge.vertex_ids
            pos1 = self.vertices[v1].position
            pos2 = self.vertices[v2].position
            if edges and not edges_id:
                plt.plot([pos1[0], pos2[0]], [pos1[1], pos2[1]], 'g-')
            if edges_id:
                mid = 0.5 * (pos1 + pos2)
                plt.text(mid[0], mid[1], str(edge.id),
                        color='green', fontsize=10,
                        ha='center', va='center')


        # Vertices
        if vertices:
            all_positions = np.array([v.position for v in self.vertices.values()])
            plt.scatter(all_positions[:, 0], all_positions[:, 1], c='red', s=10)


        #Right side cut vertices
        if right_vertices and self.right_side_vertex_ids is not None:
            right_side_positions = np.array([self.vertices[vid].position for vid in self.right_side_vertex_ids])
            plt.scatter(right_side_positions[:, 0], right_side_positions[:, 1], c='yellow', s=10)


        #Boundary points
        if boundary_points:
            boundary_vertex_ids = self.get_boundary_vertex_ids()
            boundary_pos = np.array([self.vertices[vid].position for vid in boundary_vertex_ids])
            plt.scatter(boundary_pos[:, 0], boundary_pos[:, 1], c='green', s=10)
        
        #Plot fixed vertices
        if fixed_vertices:
            if len(self.fix_vertex_xy)>0:
                positions = np.array([self.vertices[vid].position for vid in self.fix_vertex_xy])
                plt.scatter(positions[:, 0], positions[:, 1], c='black', s=50)
            if len(self.fix_vertex_x)>0:
                positions = np.array([self.vertices[vid].position for vid in self.fix_vertex_x])
                plt.scatter(positions[:, 0], positions[:, 1], c='black', s=50)
            if len(self.fix_vertex_y)>0:
                positions = np.array([self.vertices[vid].position for vid in self.fix_vertex_y])
                plt.scatter(positions[:, 0], positions[:, 1], c='black', s=50)

        plt.gca().set_aspect('equal', adjustable='box')
        plt.show()



    def plot_mean_energy_force(self, folder_name="test", save=True):
        """
        Plot the evolution of mean energy and force over iterations
        Saves or displays plots of mean energy and mean force with linear and log-log scales
        
        Parameters:
        - folder_name: folder where the plot will be saved (default: "test")
        - save: whether to save the plot to disk (default: True)
        """
        self.mean_force=self.mean_force[1:]
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))

        # Mean force: Linear scale
        axes[0, 0].plot(range(1, len(self.mean_force) + 1), self.mean_force, marker='o', label='Mean force')
        axes[0, 0].set_xlabel('Iteration number', fontsize=12)
        axes[0, 0].set_ylabel('Mean force', fontsize=12)
        axes[0, 0].set_title('Mean force vs. iterations ', fontsize=14)
        axes[0, 0].legend()
        axes[0, 0].grid(True, linestyle='--', linewidth=0.5)

        # Mean force: Log-log scale
        axes[0, 1].loglog(range(1, len(self.mean_force) + 1), self.mean_force, marker='o', label='Mean force')
        axes[0, 1].set_xlabel('Iteration number (log scale)', fontsize=12)
        axes[0, 1].set_ylabel('Mean force (log scale)', fontsize=12)
        axes[0, 1].set_title('Mean force vs. iterations (log-log scale)', fontsize=14)
        axes[0, 1].legend()
        axes[0, 1].grid(True, which="both", linestyle='--', linewidth=0.5)

        # Mean energy: Linear scale
        axes[1, 0].plot(range(1, len(self.mean_energy) + 1), self.mean_energy, marker='o', label='Mean energy', color='r')
        axes[1, 0].set_xlabel('Iteration number', fontsize=12)
        axes[1, 0].set_ylabel('Mean energy', fontsize=12)
        axes[1, 0].set_title('Mean energy vs. iterations ', fontsize=14)
        axes[1, 0].legend()
        axes[1, 0].grid(True, linestyle='--', linewidth=0.5)

        # Mean energy: Log-log scale
        axes[1, 1].loglog(range(1, len(self.mean_energy) + 1), self.mean_energy, marker='o', label='Mean energy', color='r')
        axes[1, 1].set_xlabel('Iteration number (log scale)', fontsize=12)
        axes[1, 1].set_ylabel('Mean energy (log scale)', fontsize=12)
        axes[1, 1].set_title('Mean energy vs. iterations (log-log scale)', fontsize=14)
        axes[1, 1].legend()
        axes[1, 1].grid(True, which="both", linestyle='--', linewidth=0.5)

        # Adjust layout
        plt.tight_layout()

        # Save or show the plot
        if save:
            # Ensure the output folder is in the script's directory
            script_dir = os.path.dirname(os.path.abspath(__file__))
            output_folder = os.path.join(script_dir, folder_name)
            os.makedirs(output_folder, exist_ok=True)

            # Define the file name
            file_name = f"force_energy_{self.num_cells}cells{len(self.mean_force)}.png"
            file_path = os.path.join(output_folder, file_name)

            # Save the plot
            plt.savefig(file_path)
            plt.close()
            print(f"Plot saved to {file_path}")
        else:
            # Display the plot
            plt.show()



    def save_3d_dome(self, rho, h, theta, folder_name="test"):
        """
        Generate and save a 3D visualization of the spherical cap in the specified folder

        Parameters:
        - rho: radius of the sphere
        - h: height of the spherical cap
        - theta: central angle
        - folder_name: folder where the plot will be saved
        """
        file_name = f"{theta:.3f}theta_{self.alpha}alpha_{self.beta}beta_M3.png"
        
        # Ensure the output folder is in the script's directory
        script_dir = os.path.dirname(os.path.abspath(__file__))
        output_folder = os.path.join(script_dir, folder_name)
        os.makedirs(output_folder, exist_ok=True)



        # Paths for the files
        file_path = os.path.join(output_folder, file_name)

        # Create the figure and 3D axis
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')

        # Create the sphere
        u = np.linspace(0, 2 * np.pi, 100)
        v = np.linspace(0, np.pi, 50)
        center_z = h - rho

        # Calculate Cartesian coordinates for the sphere
        x = rho * np.outer(np.cos(u), np.sin(v))
        y = rho * np.outer(np.sin(u), np.sin(v))
        z = rho * np.outer(np.ones(np.size(u)), np.cos(v)) + center_z
        ax.plot_wireframe(x, y, z, color='black', linestyle='--', linewidth=0.5)

        # Calculate incompatibility values
        incompatibility_values = [(cell.P / np.sqrt(cell.A)) for cell in self.cells.values()]

        # Normalize incompatibility values for colormap
        vmax = 7
        norm = plt.Normalize(vmin=min(incompatibility_values), vmax=min(vmax, max(incompatibility_values)))
        cmap = plt.cm.plasma

        # Plot cells
        for cell in self.cells.values():
            positions = [self.vertices[v_id].position for v_id in cell.vertex_ids]
            positions.append(positions[0])  # Close the polygon
            positions = np.array(positions)

            # Compute incompatibility and corresponding color
            incompatibility = (cell.P / np.sqrt(cell.A))
            if incompatibility > vmax:
                color = "white"
            else:
                color = cmap(norm(incompatibility))

            # Create a Poly3DCollection for the cell and add it to the plot
            poly3d = Poly3DCollection([positions], facecolors=color, linewidths=1, edgecolors='k', alpha=0.6)
            ax.add_collection3d(poly3d)

        # Add a colorbar
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array(incompatibility_values)
        cbar = plt.colorbar(sm, ax=ax, shrink=0.6)
        cbar.set_label(f'S0')


        # Set aspect ratio
        ax.set_box_aspect([1, 1, 1])

        # Save front view (rotated slightly to the right)
        ax.view_init(elev=10, azim=50)  # For the half
        #ax.view_init(elev=10, azim=10)
        ax.set_title(f'Incompatibility values. Min: {min(incompatibility_values)}, max: {max(incompatibility_values)}')
        plt.savefig(file_path)


        # Close the plot
        plt.close()

        print(f"Plot saved to {file_path}")


    def save_3d_cone(self, gb, h, folder_name="test"):
        """
        Generate and save a 3D visualization of the cone in the specified folder

        Parameters:
        - gb: radius of the base of the cone
        - h: height of the cone
        - folder_name: folder where the plot will be saved
        """
        file_name = f"{self.theta:.3f}theta_{self.alpha}alpha_{self.beta}beta_M3.png"
        
        # Ensure the output folder is in the script's directory
        script_dir = os.path.dirname(os.path.abspath(__file__))
        output_folder = os.path.join(script_dir, folder_name)
        os.makedirs(output_folder, exist_ok=True)



        # Paths for the files
        file_path = os.path.join(output_folder, file_name)

        # Create the figure and 3D axis
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')


        # Create the cone
        u = np.linspace(0, 2 * np.pi, 100)  # Angular parameter
        v = np.linspace(0, 1, 50)          # Height parameter (scaled to 0-1)
        z = h * (v[:, np.newaxis] - 1)     # Height from apex to base (negative to positive)
        x = gb * (1 - v)[:, np.newaxis] * np.cos(u)  # Radius scales down with height
        y = gb * (1 - v)[:, np.newaxis] * np.sin(u)

        ax.plot_wireframe(x, y, z, color='black', linestyle='--', linewidth=0.5)

        # Calculate incompatibility values
        incompatibility_values = [(cell.P / np.sqrt(cell.A)) for cell in self.cells.values()]
        print(f"Min incompatibility: {min(incompatibility_values)}, Max incompatibility: {max(incompatibility_values)}")

        # Normalize incompatibility values for colormap
        vmax = 7
        norm = plt.Normalize(vmin=min(incompatibility_values), vmax=min(vmax, max(incompatibility_values)))
        cmap = plt.cm.plasma

        # Plot cells
        for cell in self.cells.values():
            positions = [self.vertices[v_id].position for v_id in cell.vertex_ids]
            positions.append(positions[0])  # Close the polygon
            positions = np.array(positions)

            # Compute incompatibility and corresponding color
            incompatibility = (cell.P / np.sqrt(cell.A))
            if incompatibility > vmax:
                color = "white"
            else:
                color = cmap(norm(incompatibility))

            # Create a Poly3DCollection for the cell and add it to the plot
            poly3d = Poly3DCollection([positions], facecolors=color, linewidths=1, edgecolors='k', alpha=0.6)
            ax.add_collection3d(poly3d)

        # Add a colorbar
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array(incompatibility_values)
        cbar = plt.colorbar(sm, ax=ax, shrink=0.6)
        cbar.set_label(f'S0')

        # Set aspect ratio
        ax.set_box_aspect([1, 1, 1])

        # Save front view (rotated slightly to the right)
        ax.view_init(elev=10, azim=50)  # For the half
        #ax.view_init(elev=10, azim=10)
        ax.set_title(f'Incompatibility values. Min: {min(incompatibility_values)}, max: {max(incompatibility_values)}')
        plt.savefig(file_path)

    

        # Close the plot
        plt.close()

        print(f"Plot saved to {file_path}")





    def save_flat_dome(self, theta, folder_name):
        """
        Generate and save the flatenned spherical cap in the specified folder

        Parameters:
        - theta: central angle
        - folder_name: folder where the plot will be saved
        """
        file_name = f"{theta:.3f}theta_{self.alpha}alpha_{self.beta}beta_final.png"
        # Ensure the output folder is in the script's directory
        self.update_cell_AP()
        script_dir = os.path.dirname(os.path.abspath(__file__))
        output_folder = os.path.join(script_dir, folder_name)
        os.makedirs(output_folder, exist_ok=True)

        file_path = os.path.join(output_folder, file_name)


        plt.figure(figsize=(8, 8))
        ax = plt.gca()

        # Draw the boundary circle
        circle = plt.Circle((0, 0), self.radius, color='black', linestyle='--', fill=False, linewidth=3)
        ax.add_patch(circle)

        # Normalize incompatibility values for coloring not normalized
        incompatibility_values = [(cell.P / np.sqrt(cell.A))  for cell in self.cells.values()]
        #norm = Normalize(vmin=min(incompatibility_values), vmax=max(incompatibility_values))
        vmax=7
        norm = Normalize(vmin=min(incompatibility_values), vmax=min(vmax, max(incompatibility_values)))
        cmap = cm.plasma  # Choose a colormap


        # Plot each cell as a filled polygon (filled with not normalized incompatibility)
        for cell in self.cells.values():
            positions = [self.vertices[v_id].position for v_id in cell.vertex_ids]
            x, y = zip(*positions) 
            incompatibility = (cell.P / np.sqrt(cell.A)) 
            color = cmap(norm(incompatibility))
            ax.fill(x, y, color=color, edgecolor='k', linewidth=1, alpha=0.6)

        # Set plot properties
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array(incompatibility_values)
        cbar = plt.colorbar(sm, ax=ax, shrink=0.6) 
        cbar.set_label(f'Incompatibility. Min: {min(incompatibility_values)}, max: {max(incompatibility_values)}') 
        ax.set_aspect('equal', adjustable='box')   

        plt.savefig(file_path)
        plt.close()  

        print(f"Plot saved to {file_path}")     


    def save_simulation(self, num_iterations=None, folder_name="test", boundary_circle=True):
        """
        Generates and saves a plot of the current simulation state with optional boundary circle

        Parameters:
        - num_iterations: number of iterations to label the saved file (default: None)
        - folder_name: folder where the plot will be saved (default: "test")
        - boundary_circle: whether to include a boundary circle in the plot (default: True)
        """
        # Ensure the output folder is in the script's directory
        script_dir = os.path.dirname(os.path.abspath(__file__))
        output_folder = os.path.join(script_dir, folder_name)
        os.makedirs(output_folder, exist_ok=True)

        # Define the file name
        if self.cut:
            str = "cut"
        else:
            str = "nocut"
        if num_iterations is None:
            file_name = f"{str}_{self.num_cells}cells_{self.alpha:.2f}alpha_{self.beta:.2f}beta_{self.gamma:.2f}gamma_{self.S_end:.3f}S0_{self.L0_end:.3f}L0_{num_iterations}.png"
        else:
            file_name = f"{self.num_cells}cells_{self.S_end:.3f}S0_{self.L0_end:.3f}L0.png"
        file_path = os.path.join(output_folder, file_name)

        # Save the plot instead of showing it
        plt.figure(figsize=(8, 8))
        ax = plt.gca()

        for cell in self.cells.values():
            positions = [self.vertices[v_id].position for v_id in cell.vertex_ids]
            positions.append(positions[0])
            polygon = Polygon(positions)

            x, y = polygon.exterior.xy
            plt.plot(x, y, 'b-')

        if boundary_circle:
            circle = plt.Circle((0, 0), self.radius, color='black', linestyle='--', fill=False, linewidth = 3)
            ax.add_patch(circle)

        # Set fixed axes limits
        plt.gca().set_aspect('equal', adjustable='box')

        plt.savefig(file_path)
        plt.close()  

        print(f"Plot saved to {file_path}")


    def save_half_flat_dome(self, theta, folder_name="test", boundary_circle=True, right_vertices=False):
        """
        Generates and saves a plot of a flat mesh, including mirrored cells for symmetry

        Parameters:
        - theta: central angle
        - folder_name: folder where the plot will be saved (default: "test")
        - boundary_circle: whether to include a boundary circle in the plot (default: True)
        - right_vertices: whether to plot right-side vertices (default: False)
        """
        file_name = f"{theta:.3f}theta_{self.alpha}alpha_{self.beta}beta_half_final.png"
        self.update_cell_AP()
        script_dir = os.path.dirname(os.path.abspath(__file__))
        output_folder = os.path.join(script_dir, folder_name)
        os.makedirs(output_folder, exist_ok=True)

        file_path = os.path.join(output_folder, file_name)

        plt.figure(figsize=(8, 8))
        ax = plt.gca()

        # Draw the boundary circle
        if boundary_circle:
            circle = plt.Circle((0, 0), self.radius, color='black', linestyle='--', fill=False, linewidth=3)
            ax.add_patch(circle)

        # Normalize incompatibility values for coloring
        incompatibility_values = [(cell.P / np.sqrt(cell.A)) for cell in self.cells.values()]
        vmax = 7
        norm = Normalize(vmin=min(incompatibility_values), vmax=min(vmax, max(incompatibility_values)))
        cmap = cm.plasma

        # Plot each cell as a filled polygon and duplicate for symmetry
        for cell in self.cells.values():
            positions = [self.vertices[v_id].position for v_id in cell.vertex_ids]
            x, y = zip(*positions)
            incompatibility = (cell.P / np.sqrt(cell.A))
            color = cmap(norm(incompatibility))

            # Original cell
            ax.fill(x, y, color=color, edgecolor='k', linewidth=1, alpha=0.6)

            # Mirrored cell
            mirrored_y = [-coord for coord in y]
            ax.fill(x, mirrored_y, color=color, edgecolor='k', linewidth=1, alpha=0.6)

        # Right-side cut vertices (and mirrored)
        if right_vertices and self.right_side_vertex_ids is not None:
            right_side_positions = np.array([self.vertices[vid].position for vid in self.right_side_vertex_ids])
            plt.scatter(right_side_positions[:, 0], right_side_positions[:, 1], c='red', s=10)


        # Set plot properties
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array(incompatibility_values)
        cbar = plt.colorbar(sm, ax=ax, shrink=0.6)
        cbar.set_label(f'Incompatibility. Min: {min(incompatibility_values)}, max: {max(incompatibility_values)}')
        ax.set_aspect('equal', adjustable='box')

        plt.savefig(file_path)
        plt.close()

        print(f"Plot saved to {file_path}")


    def save_half_simulation(self, num_iterations=None, folder_name="test", boundary_circle=True):
        """
        Generates and saves a plot of the current simulation state with mirrored cells for symmetry

        Parameters:
        - num_iterations: number of iterations to label the saved file (default: None)
        - folder_name: folder where the plot will be saved (default: "test")
        - boundary_circle: whether to include a boundary circle in the plot (default: True)
        """
        script_dir = os.path.dirname(os.path.abspath(__file__))
        output_folder = os.path.join(script_dir, folder_name)
        os.makedirs(output_folder, exist_ok=True)


        # Define the file name
        if self.cut:
            str = "cut"
        else:
            str = "nocut"
        if num_iterations is None:
            file_name = f"{str}_{self.num_cells}cells_{self.alpha:.2f}alpha_{self.beta:.2f}beta_{self.gamma:.2f}gamma_{self.S_end:.3f}S0_{self.L0_end:.3f}L0_{num_iterations}.png"
        else:
            file_name = f"{self.num_cells}cells_{self.S_end:.3f}S0_{self.L0_end:.3f}L0.png"
        file_path = os.path.join(output_folder, file_name)

        plt.figure(figsize=(8, 8))
        ax = plt.gca()

        for cell in self.cells.values():
            positions = [self.vertices[v_id].position for v_id in cell.vertex_ids]
            positions.append(positions[0])
            polygon = Polygon(positions)

            x, y = polygon.exterior.xy
            plt.plot(x, y, 'b-')

            # Mirrored cell
            mirrored_y = [-coord for coord in y]
            plt.plot(x, mirrored_y, 'b-')

        if boundary_circle:
            circle = plt.Circle((0, 0), self.radius, color='black', linestyle='--', fill=False, linewidth=3)
            ax.add_patch(circle)

        ax.set_aspect('equal', adjustable='box')

        plt.savefig(file_path)
        plt.close()

        print(f"Plot saved to {file_path}")



    def get_z_coordinate_dome(self, rho, h, mesh):
        """
        Compute and assign z-coordinates for vertices on a spherical cap
        Identifies and removes cells that do not fit within the spherical cap

        Parameters:
        - rho: radius of the sphere
        - h: height of the spherical cap
        - mesh: flat mesh associated with the dome for cell removal (M1)
        """
        cells_to_kill = []
        for vertex in self.vertices.values():
            x, y = vertex.position
            sqr = rho**2 - x**2 - y**2
            if sqr < 0:
                for id in vertex.cell_ids:
                    if isinstance(id, int):
                        cells_to_kill.append(id)
            else:
                z = np.sqrt(sqr) + h - rho
            vertex.position = np.append(vertex.position, z)
        self.kill_cells(cells_to_kill)
        mesh.kill_cells(cells_to_kill)


    def get_z_coordinate_cone(self, gb, h):
        """
        Calculates the z-coordinate for each vertex to conform to a cone shape

        Parameters:
        - gb: base radius of the cone
        - h: height of the cone
        """
        for vertex in self.vertices.values():
            x, y = vertex.position
            z = -np.sqrt(x**2 + y**2) *(h/gb)
            vertex.position = np.append(vertex.position, z)



    def kill_cells(self, cell_ids):
        """
        Remove specified cells and update associated vertices and edges

        Parameters:
        - cell_ids: ids of the cells to be removed. Can be integer or a list
        """
        if isinstance(cell_ids, int):
            cell_ids = [cell_ids]

        # 1) Remove the cells
        for cid in cell_ids:
            if cid in self.cells:
                del self.cells[cid]

        # 2) Update edges
        for edge_key in list(self.edges.keys()):
            edge = self.edges[edge_key]
            # Remove deleted cell IDs
            edge.cell_ids = [c for c in edge.cell_ids if c not in cell_ids]
            # If no cells reference this edge any more, delete it
            if not edge.cell_ids:
                del self.edges[edge_key]

        # 3) Update vertices
        for vid in list(self.vertices.keys()):
            vertex = self.vertices[vid]
            # Remove deleted cell IDs
            vertex.cell_ids = [c for c in vertex.cell_ids if c not in cell_ids]
            # If no cells reference this vertex any more, delete it
            if not vertex.cell_ids:
                del self.vertices[vid]
        self.num_cells = len(self.cells)      


        #print(f"Cells {cell_ids} removed")


    def plot_3d_sphere(self, rho, h, vertices=False):
        """
        Visualize the spherical cap on a sphere

        Parameters:
        - rho: radius of the sphere
        - h: height of the spherical cap
        - vertices: whether to display the vertices (default: False)
        """
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')

        u = np.linspace(0, 2 * np.pi, 100)
        v = np.linspace(0, np.pi, 50)

        center_z = h - rho  # Adjust the sphere center along the z-axis

        # Calculate Cartesian coordinates
        x = rho * np.outer(np.cos(u), np.sin(v))
        y = rho * np.outer(np.sin(u), np.sin(v))
        z = rho * np.outer(np.ones(np.size(u)), np.cos(v)) + center_z  # Shift center in z-axis

        # Plot the sphere
        ax.plot_wireframe(x, y, z, color='black', linestyle='--', linewidth=0.5)


        # Cells
        for cell in self.cells.values():
            positions = [self.vertices[v_id].position for v_id in cell.vertex_ids]
            positions.append(positions[0])  # Close the polygon
            positions = np.array(positions)

            # Plot the cell
            ax.plot(positions[:, 0], positions[:, 1], positions[:, 2], 'b-')


        # Vertices
        if vertices:
            all_positions = np.array([v.position for v in self.vertices.values()])
            ax.scatter(all_positions[:, 0], all_positions[:, 1], all_positions[:, 2], c='red', s=10)

        # Set aspect ratio
        ax.set_box_aspect([1, 1, 1])  # Equal aspect ratio in 3D

        plt.show()


    def plot_3d_cone(self, gb, h, vertices=False):
        """
        Visualize the cone

        Parameters:
        - gb: base radius of the cone
        - h: height of the cone
        - vertices: whether to display the vertices (default: False)
        """
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')

        # Create cone
        u = np.linspace(0, 2 * np.pi, 100)  # Angular parameter
        v = np.linspace(0, 1, 50)          # Height parameter (scaled to 0-1)

        # Calculate Cartesian coordinates for the cone
        x = gb * (1 - v)[:, np.newaxis] * np.cos(u)  # Radius scales down with height
        y = gb * (1 - v)[:, np.newaxis] * np.sin(u)
        z = h * ( v[:, np.newaxis]-1)  # Starts at 0 (apex) and ends at h (base)

        # Plot the cone
        ax.plot_wireframe(x, y, z, color='black', linestyle='--', linewidth=0.5)

        # Cells (if applicable in your use case)
        for cell in self.cells.values():
            positions = [self.vertices[v_id].position for v_id in cell.vertex_ids]
            positions.append(positions[0])  # Close the polygon
            positions = np.array(positions)

            # Plot the cell
            ax.plot(positions[:, 0], positions[:, 1], positions[:, 2], 'b-')

        # Vertices
        if vertices:
            all_positions = np.array([v.position for v in self.vertices.values()])
            ax.scatter(all_positions[:, 0], all_positions[:, 1], all_positions[:, 2], c='red', s=10)

        # Set aspect ratio
        ax.set_box_aspect([1, 1, 1])  # Equal aspect ratio in 3D

        plt.show()


    def plot_incompatibility(self):
        """
        Visualize real incompatibility of cells
        """
        plt.figure(figsize=(8, 8))
        ax = plt.gca()

        # Draw the boundary circle
        circle = plt.Circle((0, 0), self.radius, color='black', linestyle='--', fill=False, linewidth=3)
        ax.add_patch(circle)

        # Normalize incompatibility values for coloring not normalized
        incompatibility_values = [(cell.P / np.sqrt(cell.A))  for cell in self.cells.values()]
        vmax=min(7, max(incompatibility_values))
        norm = Normalize(vmin=min(incompatibility_values), vmax=vmax)
        cmap = cm.plasma  # Choose a colormap

        print(f"Min incompatibility: {min(incompatibility_values)}, Max incompatibility: {max(incompatibility_values)}")

        # Plot each cell as a filled polygon (filled with not normalized incompatibility)
        for cell in self.cells.values():
            positions = [self.vertices[v_id].position for v_id in cell.vertex_ids]
            x, y = zip(*positions) 
            incompatibility = (cell.P / np.sqrt(cell.A)) 
            if incompatibility>vmax:
                color="white"
            else:
                color = cmap(norm(incompatibility))
            ax.fill(x, y, color=color, edgecolor='k', linewidth=1, alpha=0.6)

        """#Right side cut vertices
        if self.right_side_vertex_ids is not None:
            right_side_positions = np.array([self.vertices[vid].position for vid in self.right_side_vertex_ids])
            plt.scatter(right_side_positions[:, 0], right_side_positions[:, 1], c='red', s=10)"""

        # Plot vertices
        #all_positions = np.array([v.position for v in self.vertices.values()])
        #ax.scatter(all_positions[:, 0], all_positions[:, 1], c='red', s=10, label='Vertices')

        #ax.axis('off')


        # Set plot properties
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array(incompatibility_values)
        cbar = plt.colorbar(sm, ax=ax, shrink=0.6) 
        cbar.set_label('Incompatibility (S)') 
        ax.set_aspect('equal', adjustable='box')
        #ax.legend()
        plt.show()


    def plot_3d_incompatibility_dome(self, rho, h):
        """
        Visualize the incompatibility of cells on the spherical cap

        Parameters:
        - rho: radius of the sphere
        - h: height of the spherical cap
        """
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')

        # Create the sphere
        u = np.linspace(0, 2 * np.pi, 100)
        v = np.linspace(0, np.pi, 50)
        center_z = h - rho

        # Calculate Cartesian coordinates for the sphere
        x = rho * np.outer(np.cos(u), np.sin(v))
        y = rho * np.outer(np.sin(u), np.sin(v))
        z = rho * np.outer(np.ones(np.size(u)), np.cos(v)) + center_z
        ax.plot_wireframe(x, y, z, color='black', linestyle='--', linewidth=0.5)

        # Calculate incompatibility values not normalized by self.cst_S0
        incompatibility_values = [(cell.P / np.sqrt(cell.A))  for cell in self.cells.values()]
        print(f"Min incompatibility: {min(incompatibility_values)}, Max incompatibility: {max(incompatibility_values)}")

        # Normalize incompatibility values for colormap
        #norm = plt.Normalize(vmin=min(incompatibility_values), vmax=max(incompatibility_values))
        vmax=min(7, max(incompatibility_values))
        norm = plt.Normalize(vmin=min(incompatibility_values), vmax=vmax)  #to see the gradient of incompatibility. cells with really high values are too small to be really seen
        cmap = plt.cm.plasma
        

        # Plot cells
        for cell in self.cells.values():
            positions = [self.vertices[v_id].position for v_id in cell.vertex_ids]
            positions.append(positions[0])  # Close the polygon
            positions = np.array(positions)

            # Compute incompatibility and corresponding color not normalized
            incompatibility = (cell.P / np.sqrt(cell.A)) 
            if incompatibility>vmax:
                color = "white"
            else:
                color = cmap(norm(incompatibility))


            # Create a Poly3DCollection for the cell and add it to the plot
            poly3d = Poly3DCollection([positions], facecolors=color, linewidths=1, edgecolors='k', alpha=0.6)
            ax.add_collection3d(poly3d)

        # Add a colorbar
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array(incompatibility_values)
        cbar = plt.colorbar(sm, ax=ax, shrink=0.6)
        cbar.set_label('Incompatibility (S)')

        # Set aspect ratio and display
        ax.set_box_aspect([1, 1, 1])  # Equal aspect ratio in 3D
        plt.show()
    

    def plot_3d_incompatibility_cone(self, gb, h):
        """
        Visualize the incompatibility of cells of a cone

        Parameters:
        - gb: base radius of the cone
        - h: height of the cone
        """
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')

        # Create the cone
        u = np.linspace(0, 2 * np.pi, 100)  # Angular parameter
        v = np.linspace(0, 1, 50)          # Height parameter (scaled to 0-1)
        z = h * (v[:, np.newaxis] - 1)     # Height from apex to base (negative to positive)
        x = gb * (1 - v)[:, np.newaxis] * np.cos(u)  # Radius scales down with height
        y = gb * (1 - v)[:, np.newaxis] * np.sin(u)

        ax.plot_wireframe(x, y, z, color='black', linestyle='--', linewidth=0.5)

        # Calculate incompatibility values not normalized by self.cst_S0
        incompatibility_values = [(cell.P / np.sqrt(cell.A)) for cell in self.cells.values()]
        print(f"Min incompatibility: {min(incompatibility_values)}, Max incompatibility: {max(incompatibility_values)}")

        # Normalize incompatibility values for colormap
        vmax=min(6, max(incompatibility_values))
        norm = plt.Normalize(vmin=min(incompatibility_values), vmax=vmax)  #to see the gradient of incompatibility. cells with really high values are too small to be really seen
        cmap = plt.cm.plasma

        # Plot cells
        for cell in self.cells.values():
            positions = [self.vertices[v_id].position for v_id in cell.vertex_ids]
            positions.append(positions[0])  # Close the polygon
            positions = np.array(positions)

            # Compute incompatibility and corresponding color not normalized
            incompatibility = (cell.P / np.sqrt(cell.A)) 
            color = cmap(norm(incompatibility))

            # Create a Poly3DCollection for the cell and add it to the plot
            poly3d = Poly3DCollection([positions], facecolors=color, linewidths=1, edgecolors='k', alpha=0.6)
            ax.add_collection3d(poly3d)

        # Add a colorbar
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array(incompatibility_values)
        cbar = plt.colorbar(sm, ax=ax, shrink=0.6)
        cbar.set_label('Incompatibility (S)')

        # Set aspect ratio and display
        ax.set_box_aspect([1, 1, 1])  # Equal aspect ratio in 3D
        plt.show()


    def get_boundary_vertex_ids_right_of_cut(self, tol=1e-6):
        """
        Identify boundary vertices located to the right side of the cut
        Stores the list of identified vertex ids in `self.right_side_vertex_ids`

        Parameters:
        - tol: tolerance for considering a vertex to be on the outer circle (default 1e-6)
        """
        boundary_vertex_ids_right = set()  # avoid duplicates

        # 1) Identify boundary edges
        for edge in self.edges.values():
            if len(edge.cell_ids) == 1:  # true boundary edge
                for vid in edge.vertex_ids:
                    vpos = self.vertices[vid].position
                    x, y = vpos

                    # 2) must lie to the right of the cut line
                    if x <= 0:
                        continue

                    # 3) must NOT lie on the outer circle
                    dist = np.linalg.norm(vpos)
                    if abs(dist - self.radius) < tol:
                        continue

                    boundary_vertex_ids_right.add(vid)

        # store and return
        self.right_side_vertex_ids = list(boundary_vertex_ids_right)


    def get_boundary_vertex_ids(self):
        """
        Identify all boundary vertices and mark boundary cells

        Returns:
        - List of unique boundary vertex ids
        """
        boundary_vertex_ids = set()  # To store unique boundary vertices

        # Step 1: Identify boundary edges and add their vertices directly
        for edge in self.edges.values():
            if len(edge.cell_ids) == 1:  # Boundary edge
                boundary_vertex_ids.update(edge.vertex_ids)

        # Step 2: Mark all cells that contain any of these boundary vertices
        for cell in self.cells.values():
            # If any vertex of this cell is a boundary vertex
            if any(vid in boundary_vertex_ids for vid in cell.vertex_ids):
                cell.is_boundary = True


        return list(boundary_vertex_ids)
    

    def random_noise(self, displacement=0.2):
        """
        Apply random noise to the positions of interior vertices
        Ensure that vertices remain within the system's circular boundary
        Updates cell area and perimeter after perturbation

        Parameters:
        - displacement: maximum magnitude of random displacement applied to each vertex (default 0.2)
        """
        # Get the list of boundary vertex IDs
        boundary_vertex_ids = self.get_boundary_vertex_ids()

        # Collect all vertex positions into an array
        all_positions = np.array([v.position for v in self.vertices.values()])

        # Identify interior vertex IDs (not in boundary)
        interior_indices = [
            idx for idx, vertex_id in enumerate(self.vertices.keys()) 
            if vertex_id not in boundary_vertex_ids
        ]

        # Create perturbations for interior vertices
        perturbations = (np.random.rand(len(interior_indices), 2) - 0.5) * displacement

        # Apply perturbations to interior vertex positions
        for i, idx in enumerate(interior_indices):
            new_position = all_positions[idx] + perturbations[i]
            
            # Ensure that the new position is still inside the circle
            distance_from_center = np.linalg.norm(new_position)
            
            # If the distance is greater than the radius, limit the movement
            if distance_from_center > self.radius:
                new_position = new_position * (self.radius / distance_from_center)
            
            # Update the position if it's within the circle
            all_positions[idx] = new_position

        # Update vertex positions in the vertices dictionary
        for i, vertex in enumerate(self.vertices.values()):
            vertex.position = all_positions[i]

        # Update cells' area and perimeter based on new vertex positions
        for cell in self.cells.values():
            cell.update_AP(self.vertices)



    def rotate_vertices(self, angle_degrees=90):
        """
        Rotate every vertex around the origin by the specified angle (in degrees)
        Positive angles rotate counterclockwise, negative clockwise

        (x, y) -> ( x*cosθ - y*sinθ, x*sinθ + y*cosθ )

        After rotating, updates all dependent cell and edge metrics
        """
        θ = np.deg2rad(angle_degrees)
        cosθ, sinθ = np.cos(θ), np.sin(θ)

        # Apply rotation to every vertex
        for vertex in self.vertices.values():
            x, y = vertex.position
            vertex.position = np.array([
                x * cosθ - y * sinθ,
                x * sinθ + y * cosθ
            ])


    def generate_triangles(self, side_length=1):
        """
        Generate a triangular mesh based on the cut status.

        If `self.cut` is True, generates right-half only
        Otherwise, generates a full triangular mesh

        Parameters:
        - side_length: edge length of triangles
        """
        if self.cut:
            self.generate_half_hybrid_circular_mesh(side_length=side_length)
        else:
            self.generate_hybrid_circular_mesh(side_length=side_length)


    def generate_half_hybrid_circular_mesh(self, side_length=1, perturbation_scale=0.5):
        """
        Generate a triangular mesh restricted to the right half of a circular domain

        Parameters:
        - side_length: edge length
        - perturbation_scale: scale factor for interior vertex random displacement
        """
        # Initialize
        vertices = {}
        vertices_by_id = {}
        hybrid_cells = {}

        # Compute radius
        self.radius = np.sqrt((np.sqrt(3) * self.num_cells * side_length**2) / (4 * np.pi))

        # 1) Generate grid and keep only right‐half points
        x = np.arange(0, self.radius + side_length, side_length)
        y = np.arange(-self.radius, self.radius + side_length, side_length)
        xx, yy = np.meshgrid(x, y)
        pts = np.vstack((xx.ravel(), yy.ravel())).T
        interior = pts[np.linalg.norm(pts, axis=1) <= self.radius]

        # 2a) half‐circle arc
        theta = np.linspace(-np.pi/2, np.pi/2,
                            int(np.pi * self.radius / side_length),
                            endpoint=False)
        half_circle = np.c_[self.radius * np.cos(theta),
                            self.radius * np.sin(theta)]

        # 2b) cut line trimmed to circle
        y_cut = np.arange(-self.radius, self.radius + side_length, side_length)
        raw_cut = np.c_[np.zeros_like(y_cut), y_cut]
        cut_line = raw_cut[np.linalg.norm(raw_cut, axis=1) <= self.radius]

        # 2c) fixed points
        fixed_origin = np.array([[0.0, 0.0]])
        fixed_top    = np.array([[0.0, self.radius]])

        # Combine boundary
        boundary = np.vstack((half_circle, cut_line,
                            fixed_origin, fixed_top))

        # 3) Triangulate
        points   = np.vstack((interior, boundary))
        delaunay = Delaunay(points)

        # Helpers
        def add_vertex(pos):
            key = (round(pos[0],6), round(pos[1],6))
            if key not in vertices:
                vid = len(vertices) + 1
                v = Vertex(id=vid, position=pos.copy())
                vertices[key] = v
                vertices_by_id[vid] = v
            return vertices[key]

        def add_edge(vertex_ids):
            edge_key = tuple(sorted(vertex_ids))

            # si l'arête existe déjà (on la cherche par ses vertices)
            for edge in self.edges.values():
                if tuple(sorted(edge.vertex_ids)) == edge_key:
                    return edge

            # sinon création
            edge_id = len(self.edges) + 1
            edge = Edge(id=edge_id, vertex_ids=vertex_ids)

            # IMPORTANT : clé = edge.id (PAS edge_key)
            self.edges[edge_id] = edge

            return edge

        # 4) Build mesh
        for simplex in delaunay.simplices:
            verts = [add_vertex(points[i]) for i in simplex]
            vids  = [v.id for v in verts]
            cid   = len(hybrid_cells) + 1

            # edges
            for i in range(3):
                add_edge((vids[i], vids[(i + 1) % 3]), cid)

            # cell
            center = np.mean([v.position for v in verts], axis=0)
            relpos = 1 - min(1, np.linalg.norm(center) / self.radius)
            hybrid_cells[cid] = Cell(
                L0=side_length,
                id=cid, vertices=verts, num_neigh=3,
                relative_position=relpos,
                alpha=self.alpha, beta=self.beta, gamma=self.gamma,
                A0=None, P0=None
            )

        # 5) Assign to self
        self.vertices  = vertices_by_id
        self.cells     = hybrid_cells
        self.num_cells = len(hybrid_cells)

        # 6) Initial metrics
        self.update_cell_AP()
        self.update_edge_length()
        self.compute_energy()

        # vertices to track on the cut

        tol = 1e-6
        self.right_side_vertex_ids = []
        for vid, vertex in self.vertices.items():
            x, y = vertex.position
            # exact origin
            if abs(x) < tol and abs(y) < tol:
                self.right_side_vertex_ids.append(vid)
            # on the segment x=0, y in [0,-radius]
            if abs(x) < tol and (y <= tol) and (y >= - self.radius - tol):
                self.right_side_vertex_ids.append(vid)




        # 7) Capture fixed‐vertex IDs and all on x=0, 0<=y<=radius
        tol = 1e-6
        self.fix_vertex_xy = []
        self.fix_vertex_y  = []

        for vid, vertex in self.vertices.items():
            x, y = vertex.position
            # exact origin
            if abs(x) < tol and abs(y) < tol:
                self.fix_vertex_xy.append(vid)
            # on the segment x=0, y in [0,radius]
            if abs(x) < tol and (y >= -tol) and (y <= self.radius + tol):
                self.fix_vertex_y.append(vid)

        # 8) Perturb interior only
        self.random_noise(displacement=perturbation_scale * side_length)

        # 9) Recompute metrics
        self.update_cell_AP()
        self.update_edge_length()
        self.compute_energy()
        self.get_boundary_vertex_ids()
        self.rotate_vertices()
        

    def generate_hybrid_circular_mesh(self, side_length=0.1, perturbation_scale=0.5):
        """
        Generate a full triangular mesh within a circular domain

        Parameters:
        - side_length: edge length
        - perturbation_scale: scale factor for random displacement of interior vertices
        """
        # Initialize data structures
        vertices = {}
        vertices_by_id = {}
        hybrid_cells = {}

        # Calculate the radius based on mesh size and number of cells
        self.radius = np.sqrt((np.sqrt(3) * self.num_cells * side_length**2) / (4 * np.pi))

        # Create grid points
        x = np.arange(-self.radius, self.radius + side_length, side_length)
        y = np.arange(-self.radius, self.radius + side_length, side_length)
        xx, yy = np.meshgrid(x, y)
        grid_points = np.vstack((xx.ravel(), yy.ravel())).T

        # Filter points within the circle
        circular_mask = np.linalg.norm(grid_points, axis=1) <= self.radius
        interior_points = grid_points[circular_mask]

        # Generate boundary points
        theta = np.linspace(0, 2 * np.pi, int(2 * np.pi * self.radius / side_length), endpoint=False)
        boundary_points = np.c_[self.radius * np.cos(theta), self.radius * np.sin(theta)]

        # Combine interior and boundary points
        points = np.vstack((interior_points, boundary_points))

        # Perform Delaunay triangulation
        delaunay = Delaunay(points)

        # Helper functions for vertices and edges
        def add_vertex(position):
            pos_tuple = tuple(np.round(position, decimals=6))
            if pos_tuple not in vertices:
                vertex_id = len(vertices) + 1
                vertex = Vertex(id=vertex_id, position=position)
                vertices[pos_tuple] = vertex
                vertices_by_id[vertex_id] = vertex
            return vertices[pos_tuple]

        def add_edge(vertex_ids):
            edge_key = tuple(sorted(vertex_ids))

            # si l'arête existe déjà (on la cherche par ses vertices)
            for edge in self.edges.values():
                if tuple(sorted(edge.vertex_ids)) == edge_key:
                    return edge

            # sinon création
            edge_id = len(self.edges) + 1
            edge = Edge(id=edge_id, vertex_ids=vertex_ids)

            # IMPORTANT : clé = edge.id (PAS edge_key)
            self.edges[edge_id] = edge

            return edge

        # Add vertices from points
        for point in points:
            add_vertex(point)

        # Add edges and cells based on Delaunay triangulation
        for simplex in delaunay.simplices:
            simplex_vertices = [add_vertex(points[i]) for i in simplex]
            vertex_ids = [v.id for v in simplex_vertices]

            # Create edges and associate them with the current cell
            for i in range(3):
                add_edge((vertex_ids[i], vertex_ids[(i + 1) % 3]), len(hybrid_cells) + 1)

            # Calculate `relative_position` based on distance from center
            center_of_cell = np.mean([v.position for v in simplex_vertices], axis=0)
            relative_position = 1 - min(1, np.linalg.norm(center_of_cell) / self.radius)

            # Create hybrid cells
            hybrid_cell = Cell(
                L0=side_length,
                id=len(hybrid_cells) + 1,
                vertices=simplex_vertices,
                num_neigh=3,  # Triangles have 3 neighbors
                relative_position=relative_position,
                alpha=self.alpha,
                beta=self.beta,
                gamma=self.gamma,
                A0=None,
                P0=None,
                mode="triangle"
            )
            hybrid_cells[len(hybrid_cells) + 1] = hybrid_cell

        # Update object attributes
        self.vertices = vertices_by_id
        self.cells = hybrid_cells
        self.num_cells = len(hybrid_cells)

        # Compute additional properties
        self.update_cell_AP()
        self.update_edge_length()
        self.compute_energy()

        # Apply perturbations using random_noise function
        self.random_noise(displacement=perturbation_scale * side_length)

        # Compute additional properties after perturbation
        self.update_cell_AP()
        self.update_edge_length()
        self.compute_energy()
        self.get_boundary_vertex_ids()



    def three_triangles(self, side_length=1):
        """
        Create the triple triangle mesh

        Parameters:
        - side_length : length of the sides of the triangles
        """
        vertices = {}
        vertices_by_id = {}
        triangles = {}
        L0_init = side_length

        def add_vertex(position):
            pos_tuple = tuple(np.round(position, decimals=6))
            if pos_tuple not in vertices:
                vertex_id = len(vertices) + 1
                vertex = Vertex(id=vertex_id, position=position)
                vertices[pos_tuple] = vertex
                vertices_by_id[vertex_id] = vertex
            return vertices[pos_tuple]

        def add_edge(vertex_ids):
            edge_key = tuple(sorted(vertex_ids))

            # si l'arête existe déjà (on la cherche par ses vertices)
            for edge in self.edges.values():
                if tuple(sorted(edge.vertex_ids)) == edge_key:
                    return edge

            # sinon création
            edge_id = len(self.edges) + 1
            edge = Edge(id=edge_id, vertex_ids=vertex_ids)

            # IMPORTANT : clé = edge.id (PAS edge_key)
            self.edges[edge_id] = edge

            return edge

        # Calculate the heights and positions
        H = (L0_init / 2) / np.cos(np.pi / 6)
        h = (L0_init / 2) * np.tan(np.pi / 6)

        # Add vertices
        p1 = add_vertex((0, 0))   # Vertex 1
        p2 = add_vertex((0, -H))  # Vertex 2
        p3 = add_vertex((L0_init / 2, h))  # Vertex 3
        p4 = add_vertex((-L0_init / 2, h))  # Vertex 4

        # Manually define edges
        
        edge2 = add_edge(p1.id, p2.id)  # Edge shared by triangles 1 and 2
        edge3 = add_edge(p1.id, p4.id)  # Edge shared by triangles 1 and 3
        edge4 = add_edge(p1.id, p3.id)  # Edge shared by triangles 2 and 3
        edge5 = add_edge(p3.id, p4.id)  # Edge in triangle 3
        edge6 = add_edge(p2.id, p3.id)  # Edge in triangle 2

        if self.cut:
            # Duplicate vertex p2
            p2_prime = Vertex(id=len(vertices_by_id) + 1, position=p2.position)
            vertices_by_id[p2_prime.id] = p2_prime
            edge1 = add_edge(p2_prime.id, p4.id)  # Edge shared by triangle 1

            # Create new edge between p1 and p2_prime
            edge2_prime = add_edge(p1.id, p2_prime.id)

            # Update triangle1 to use p2_prime
            triangles[1] = Cell(
                id=1,
                vertices=[p1, p4, p2_prime],  # Use p2_prime
                num_neigh=2,
                relative_position=0,
                L0=self.L0_end,
                alpha=self.alpha,
                beta=self.beta,
                gamma=self.gamma,
                S0=self.S0,
                A0=None,
                P0=None,
                mode=self.mode
            )
            triangles[1].edges = [edge1, edge2_prime, edge3]

            # Update triangle2 to continue using p2
            triangles[2] = Cell(
                id=2,
                vertices=[p1, p2, p3],  # Continue using p2
                num_neigh=2,
                relative_position=0,
                L0=self.L0_end,
                alpha=self.alpha,
                beta=self.beta,
                gamma=self.gamma,
                S0=self.S0,
                A0=None,
                P0=None,
                mode=self.mode
            )
            triangles[2].edges = [edge2, edge4, edge6]

            # Triangle3 remains unchanged
            triangles[3] = Cell(
                id=3,
                vertices=[p1, p3, p4],
                num_neigh=2,
                relative_position=0,
                L0=self.L0_end,
                alpha=self.alpha,
                beta=self.beta,
                gamma=self.gamma,
                S0=self.S0,
                A0=None,
                P0=None,
                mode=self.mode
            )
            triangles[3].edges = [edge3, edge4, edge5]

            # Update edge-to-cell associations
            edge2.remove_cell_id(1)
            edge2_prime.add_cell_id(1)
        else:
            edge1 = add_edge(p2.id, p4.id)  # Edge shared by triangle 1
            # Without cutting, create the original three triangles
            triangles[1] = Cell(
                id=1,
                vertices=[p1, p4, p2],
                num_neigh=2,
                relative_position=0,
                L0=self.L0_end,
                alpha=self.alpha,
                beta=self.beta,
                gamma=self.gamma,
                S0=self.S0,
                A0=None,
                P0=None,
                mode=self.mode
            )
            triangles[1].edges = [edge1, edge2, edge3]

            triangles[2] = Cell(
                id=2,
                vertices=[p1, p2, p3],
                num_neigh=2,
                relative_position=0,
                L0=self.L0_end,
                alpha=self.alpha,
                beta=self.beta,
                gamma=self.gamma,
                S0=self.S0,
                A0=None,
                P0=None,
                mode=self.mode
            )
            triangles[2].edges = [edge2, edge4, edge6]

            triangles[3] = Cell(
                id=3,
                vertices=[p1, p3, p4],
                num_neigh=2,
                relative_position=0,
                L0=self.L0_end,
                alpha=self.alpha,
                beta=self.beta,
                gamma=self.gamma,
                S0=self.S0,
                A0=None,
                P0=None,
                mode=self.mode
            )
            triangles[3].edges = [edge3, edge4, edge5]

        # Update the class properties
        self.vertices = vertices_by_id
        self.cells = triangles
        self.update_cell_AP()
        self.update_edge_length()
        self.num_cells = len(self.cells)
        self.compute_energy()
        self.radius=0



    def copy_scale_mesh(self, scale_factor):
        """
        Create a scaled copy of the mesh

        Parameters:
        - scale_factor: factor to scale vertex positions and cell properties

        Returns:
        - new mesh instance with all vertices and cells scaled
        """
        # Create a copy of the current mesh
        scaled_mesh = Mesh(num_cells=self.num_cells)

        # Scale the radius
        scaled_mesh.radius = self.radius * scale_factor

        # Duplicate vertices and scale positions
        scaled_vertices = {}
        scaled_vertices_by_id = {}

        for vertex_id, vertex in self.vertices.items():
            scaled_position = vertex.position * scale_factor
            scaled_vertex = Vertex(id=vertex_id, position=scaled_position)
            scaled_vertices[tuple(np.round(scaled_position, decimals=6))] = scaled_vertex
            scaled_vertices_by_id[scaled_vertex.id] = scaled_vertex

        # Duplicate cells and adjust them
        scaled_cells = {}

        for cell_id, cell in self.cells.items():
            scaled_vertices_of_cell = [scaled_vertices[tuple(np.round(self.vertices[v_id].position * scale_factor, decimals=6))] for v_id in cell.vertex_ids]
            
            # Create a new scaled cell with the same properties but scaled vertices
            scaled_cell = Cell(
                id=cell_id,
                vertices=scaled_vertices_of_cell,
                num_neigh=cell.num_neighbors,
                relative_position=cell.relative_position,  # Relative position may stay the same
                alpha=cell.alpha,
                beta=cell.beta,
                gamma=cell.gamma,
                A0=cell.A0 * (scale_factor ** 2),  # Scale A0 by the square of the scale factor
                P0=cell.P0 * scale_factor,   
            )
            
            # Add the scaled cell to the new mesh
            scaled_cells[cell_id] = scaled_cell

        # Assign the scaled vertices and cells to the new mesh
        scaled_mesh.vertices = scaled_vertices_by_id
        scaled_mesh.cells = scaled_cells
        scaled_mesh.num_cells = len(scaled_cells)

        # Adjust the edges similarly if necessary
        scaled_mesh.edges = self.edges  # assuming edges will remain the same, adjust if needed


        # Recompute the properties of the scaled mesh
        scaled_mesh.update_cell_AP()
        scaled_mesh.update_edge_length()
        scaled_mesh.compute_energy()



        scaled_mesh.dt = self.dt
        scaled_mesh.mode = self.mode
        scaled_mesh.S_end = self.S_end
        scaled_mesh.L0_end = self.L0_end
        scaled_mesh.alpha = self.alpha
        scaled_mesh.beta = self.beta
        scaled_mesh.gamma = self.gamma
        scaled_mesh.cut = self.cut
        scaled_mesh.L0 = self.L0
        scaled_mesh.cst_S0 = self.cst_S0
        scaled_mesh.S0 = self.S0
        scaled_mesh.total_force = self.total_force
        scaled_mesh.mean_force = self.mean_force
        scaled_mesh.total_energy = self.total_energy
        scaled_mesh.mean_energy = self.mean_energy
        scaled_mesh.grad_S = self.grad_S
        scaled_mesh.grad_L0 = self.grad_L0
        scaled_mesh.grad_mode = self.grad_mode
        scaled_mesh.theta = self.theta
        scaled_mesh.right_side_vertex_ids = self.right_side_vertex_ids
        scaled_mesh.fix_vertex_xy = self.fix_vertex_xy
        scaled_mesh.fix_vertex_x = self.fix_vertex_x
        scaled_mesh.fix_vertex_y = self.fix_vertex_y




        return scaled_mesh


    def global_S0(self, gb):
        """
        Compute a global shape parameter S0 for the spherical cap

        Parameters:
        - gb: radius of the spherical cap

        Returns:
        - calculated global S0 value
        """
        self.update_cell_AP()
        total_area = 0
        total_perimeter = 0
        sum_S0 = 0
        for edge in self.edges.values():
            if len(edge.cell_ids)==1:
                total_perimeter += edge.L
        for cell in self.cells.values():
            total_area += cell.A
            sum_S0 += cell.P/np.sqrt(cell.A)
        if self.mode=="triangle" and self.cut:
            global_S0 = (2*np.pi*gb)/np.sqrt(2*total_area)  #for the half sphere
        else:
            global_S0 = (2*np.pi*gb)/np.sqrt(total_area)    # for a total sphere
        return global_S0



    def save_plot_with_fit(self, x_curve, a, b, global_S0, folder_name, right_vertices=True):
        """
        Save a plot of the mesh cells with the fitted curve

        Parameters:
        - x_curve: array of x values for the fit curve
        - a, b: parameters for the function y = a * x^b
        - global_S0: add the value of global S0 to the name
        - folder_name: name of folder to save the plot
        - right_vertices: wether to plot right side vertices on the plot (default: True)
        """
        file_name = f"global_S0_{global_S0}.png"
        # Ensure the output folder is in the script's directory
        self.update_cell_AP()
        script_dir = os.path.dirname(os.path.abspath(__file__))
        output_folder = os.path.join(script_dir, folder_name)
        os.makedirs(output_folder, exist_ok=True)

        file_path = os.path.join(output_folder, file_name)


        plt.figure(figsize=(8, 8))

        # Draw the boundary circle
        circle = plt.Circle((0, 0), self.radius, color='black', linestyle='--', fill=False, linewidth=3)
        plt.gca().add_patch(circle)


        # Cells
        for cell in self.cells.values():
            positions = [self.vertices[v_id].position for v_id in cell.vertex_ids]
            positions.append(positions[0])  # Close the polygon
            polygon = Polygon(positions)

            # Plot the cell
            x, y = polygon.exterior.xy
            plt.plot(x, y, 'b-')

        # Generate the function curve using x, a, and b
        y_curve = a * (x_curve ** b)

        # Plot the fit curve
        plt.plot(x_curve, y_curve, label=f'Fit: f(x)={a:.3f}*x^{b:.3f}', color='red')

            

        #Right side cut vertices
        if right_vertices and self.right_side_vertex_ids is not None:
            right_side_positions = np.array([self.vertices[vid].position for vid in self.right_side_vertex_ids])
            plt.scatter(right_side_positions[:, 0], right_side_positions[:, 1], c='red', s=10)


        plt.gca().set_aspect('equal', adjustable='box')
        plt.title(f"Cells with fit curve, global S0 = {global_S0}")
        #plt.show()
        plt.savefig(file_path)
        plt.close()  




    def get_AP_from_other_mesh(self, mesh):
        """
        Copy target areas (A0) and perimeters (P0) from another mesh A and P

        Parameters:
        - mesh: another Mesh instance to copy data from
        """
        for id in self.cells.keys():
            self.cells[id].A0 = mesh.cells[id].A       #A from M3
            self.cells[id].P0 = mesh.cells[id].P       #P from M3


    def get_AM1_from_other_mesh(self, mesh):
        """
        Update target perimeter P0 based on incompatibility from another mesh

        Parameters:
        - mesh: another Mesh instance to reference
        """
        for id in self.cells.keys():
            self.cells[id].P0 = (mesh.cells[id].P / np.sqrt(mesh.cells[id].A))*np.sqrt(self.cells[id].A0) #compute P0 with the incompatibility from M3


    def get_AM2_from_other_mesh(self, mesh):
        """
        Update target area A0 and perimeter P0 with incompatibility from another mesh

        Parameters:
        - mesh: another Mesh instance to reference
        """
        for id in self.cells.keys():
            self.cells[id].A0 = mesh.cells[id].A0   #A from M2 because of the gradual rho
            self.cells[id].P0 = (mesh.cells[id].P / np.sqrt(mesh.cells[id].A))*np.sqrt(self.cells[id].A0) #compute P0 with the incompatibility from M3


    def get_AM3_from_other_mesh(self, mesh):
        """
        Update target area A0 and perimeter P0 from another mesh with incompatibility

        Parameters:
        - mesh: another Mesh instance to reference
        """
        for id in self.cells.keys():
            self.cells[id].A0 = mesh.cells[id].A       #A from M3
            self.cells[id].P0 = (mesh.cells[id].P / np.sqrt(mesh.cells[id].A))*np.sqrt(self.cells[id].A0) #compute P0 with the incompatibility from M3


    def get_3d_AP(self, h, L):
        """
        Compute target area and perimeter for cells assuming a tetrahedron

        Parameters:
        - h: height parameter for the tetrahedron
        - L: base length parameter for the tetrahedron
        """
        for cell in self.cells.values():
            cell.A0 = (L/2)*np.sqrt((L**2/12) + h**2)
            cell.P0 = 2*np.sqrt((L**2/3) + h**2) + L


    def compute_opening_angle(self, h, L):
        """
        Calculate the opening angle formed by three vertices for the tetrahedron case
        Print both analytical and real value

        Parameters:
        - h: height of the tetrahedron
        - L: base length of the tetrahedron

        Returns:
        - opening_angle: computed opening angle in degrees
        """
        # Extract positions as NumPy arrays
        v1 = np.array(self.vertices[1].position, dtype=np.float64)
        v2 = np.array(self.vertices[2].position, dtype=np.float64)
        v3 = np.array(self.vertices[3].position, dtype=np.float64)

        # Vectors from v1 to v2 and v1 to v3
        vec1 = v2 - v1
        vec2 = v3 - v1

        # Dot product and magnitudes
        dot_product = np.dot(vec1, vec2)
        magnitude_vec1 = np.linalg.norm(vec1)
        magnitude_vec2 = np.linalg.norm(vec2)

        # Ensure magnitudes are non-zero
        if magnitude_vec1 == 0 or magnitude_vec2 == 0:
            raise ValueError("One of the vectors has zero length, angle cannot be determined.")

        # Cosine of the angle, clamped to avoid numerical errors
        cos_theta = np.clip(dot_product / (magnitude_vec1 * magnitude_vec2), -1, 1)

        # Angle in degrees
        angle_degrees = np.degrees(np.arccos(cos_theta))
        num = (h**2 - (L**2)/6)
        den = ((L**2)/3 + h**2)
        theoretical_angle_degrees = np.degrees(np.arccos(num/den))
        

        # Specific opening angle
        opening_angle = 360 - 3 * angle_degrees
        theoretical_opening_angle = 360 - 3 * theoretical_angle_degrees
        print(f"Analytical opening angle: {round(theoretical_opening_angle, 10)}")
        print(f"Actual opening angle: {round(opening_angle, 10)}")

        # Round to an appropriate precision
        return round(opening_angle, 6)






    def check_T1(self):
        """
        Check the length of each edge and perform T1 if needed
        """
        for edge in self.edges.values():
            if edge.L < self.T1_thr:
                self.T1_transition(edge_id=edge.id)


    def T1_transition(self, edge_id):
        """
        Perform a T1 transition 

        Process:
        - identifies the edge to be flipped
        - finds the surrounding cells
        - rotates the edge vertices by 90 degrees (anticlockwise direction)
        - updates local cell connectivity
        - rebuilds the global edge list to ensure consistency

        Parameters:
        - edge_id: id of the edge undergoing the T1 transition
        """

        Medge = self.edges[edge_id]
        if len(Medge.vertex_ids) != 2:
            return

        # Identifier les vertices
        v1_id, v2_id = Medge.vertex_ids
        v1 = self.vertices[v1_id]
        v2 = self.vertices[v2_id]

        # Identifier les cellules
        list_cid = np.unique(np.concatenate((v1.cell_ids, v2.cell_ids)))
        nb_cells = len(list_cid)

        c1_id = next((x for x in v1.cell_ids if x not in v2.cell_ids), None)
        c2_id = next((x for x in v2.cell_ids if x not in v1.cell_ids), None)

        c1 = self.cells[c1_id] if c1_id is not None else None
        c2 = self.cells[c2_id] if c2_id is not None else None

        remaining = [x for x in list_cid if x not in (c1_id, c2_id)]
        c3_id = remaining[0] if len(remaining) > 0 else None
        c4_id = remaining[1] if len(remaining) > 1 else None

        c3 = self.cells[c3_id] if c3_id is not None else None
        c4 = self.cells[c4_id] if c4_id is not None else None

        if nb_cells >= 3:
            centers = []
            for c in [c1, c2, c3, c4]:
                if c is not None:
                    centers.append(c.center_of_cell(self.vertices))

            if not is_anticlockwise(centers):
                c3_id, c4_id = c4_id, c3_id
                c3, c4 = c4, c3

            # Rotation à 90 degrés sens anti horaire
            center_edge = (v1.position + v2.position) / 2
            theta = np.pi / 2

            R = np.array([
                [np.cos(theta), -np.sin(theta)],
                [np.sin(theta),  np.cos(theta)]
            ])

            v1.position = center_edge + R @ (v1.position - center_edge)
            v2.position = center_edge + R @ (v2.position - center_edge)


            # Mise a jour topologie
            # c1 : ajouter v2 après v1
            if c1 is not None and v1_id in c1.vertex_ids:
                i = c1.vertex_ids.index(v1_id)
                c1.vertex_ids.insert(i + 1, v2_id)

            # c2 : ajouter v1 après v2
            if c2 is not None and v2_id in c2.vertex_ids:
                i = c2.vertex_ids.index(v2_id)
                c2.vertex_ids.insert(i + 1, v1_id)

            # c3 : retirer v2
            if c3 is not None and v2_id in c3.vertex_ids:
                c3.vertex_ids.remove(v2_id)

            # c4 : retirer v1
            if c4 is not None and v1_id in c4.vertex_ids:
                c4.vertex_ids.remove(v1_id)

            # v1 : supprimer c4_id et ajouter c2_id 
            if c4_id is not None:
                v1.cell_ids = [x for x in v1.cell_ids if x != c4_id]
            if c2_id is not None:
                if c2_id not in v1.cell_ids:
                    v1.cell_ids.append(c2_id)

            # v2 : supprimer c3_id et ajouter c1_id
            if c3_id is not None:
                v2.cell_ids = [x for x in v2.cell_ids if x != c3_id]
            if c1_id is not None:
                if c1_id not in v2.cell_ids:
                    v2.cell_ids.append(c1_id)
        
        else:
            print(f" Cas {nb_cells} non géré pour une T1 ")
            return

        # Autres updates
        self.clean_T1_edges(v1_id, v2_id, c1_id, c2_id, c3_id, c4_id)
        
        # Mise à jour locale des voisins et du statut de bordure
        #for cid in (c1_id, c2_id, c3_id, c4_id):
            #self.cells[cid].update_neighbors_and_boundary(self.edges)

        self.update_cell_AP()
        self.update_edge_length()

        self.update_cell_AP()
        self.update_edge_length()
        

    def clean_T1_edges(self, v1_id, v2_id, c1_id, c2_id, c3_id, c4_id):
        """
        Remove the 4 old edges, and create the 4 new ones
        """

        # 1. Mise à jour de l'arête (v1, v2)
        v1v2_edge = None
        for eid, edge in self.edges.items():
            if set(edge.vertex_ids) == {v1_id, v2_id}:
                v1v2_edge = edge
                break

        if v1v2_edge is None:
            raise ValueError("L'arête v1-v2 est introuvable.")

        # L'arête v1-v2 devient partagée uniquement par c1 et c2
        v1v2_edge.cell_ids = [c1_id, c2_id]

        # 2. Supprimer toutes les autres arêtes incidentes à v1 ou v2
        edges_to_remove = []
        for eid, edge in self.edges.items():
            if edge.id == v1v2_edge.id:
                continue
            if v1_id in edge.vertex_ids or v2_id in edge.vertex_ids:
                edges_to_remove.append(eid)

        for eid in edges_to_remove:
            del self.edges[eid]

        # 3. Récupérer les voisins dans les nouvelles listes de c3 et c4
        c3_verts = self.cells[c3_id].vertex_ids
        c4_verts = self.cells[c4_id].vertex_ids

        def get_neighbors(cell_list, vid):
            n = len(cell_list)
            idx = cell_list.index(vid)
            prev = cell_list[(idx - 1) % n]
            nxt = cell_list[(idx + 1) % n]
            return prev, nxt

        prev_v1, next_v1 = get_neighbors(c3_verts, v1_id)
        prev_v2, next_v2 = get_neighbors(c4_verts, v2_id)

        # 4. Créer les nouvelles arêtes
        #    Générer le prochain id disponible
        next_id = max(self.edges.keys(), default=-1) + 1

        #    Fonction utilitaire : retourne les cellules qui contiennent deux sommets
        def cells_containing(a, b):
            return [cid for cid, cell in self.cells.items()
                    if a in cell.vertex_ids and b in cell.vertex_ids]

        new_edges = []
        # Pour v1
        for neighbor in (prev_v1, next_v1):
            e = Edge(next_id, (v1_id, neighbor))
            e.cell_ids = cells_containing(v1_id, neighbor)
            new_edges.append(e)
            next_id += 1

        # Pour v2
        for neighbor in (prev_v2, next_v2):
            e = Edge(next_id, (v2_id, neighbor))
            e.cell_ids = cells_containing(v2_id, neighbor)
            new_edges.append(e)
            next_id += 1

        # 5. Ajouter les nouvelles arêtes au dictionnaire
        for e in new_edges:
            self.edges[e.id] = e

        # 6. Recalculer les longueurs
        for edge in self.edges.values():
            p1 = self.vertices[edge.vertex_ids[0]].position
            p2 = self.vertices[edge.vertex_ids[1]].position
            edge.L = np.linalg.norm(p1 - p2)






    ### relaxation_steps pour uniformiser la taille des cellules
    ## seed et clip_to_circle à passer en paramètre
    def generate_voronoi(self, side_length=1, A0=None, P0=None, generator_factor=3, relaxation_steps=10, seed=None, clip_to_circle=True):
        """
        Génére un maillage de Voronoï circulaire avec exactement num_cells cellules.
        Si relaxation_steps > 0, applique une relaxation de Lloyd pour uniformiser
        la taille des cellules.
        """
        if seed is not None:
            np.random.seed(seed)

        # --- Cibles identiques au mode "circle" ---
        if A0 is None:
            A0_cell = np.pi * side_length**2
        else:
            A0_cell = A0

        S0_circle = 2.0 * np.sqrt(np.pi)
        S0_cell = self.S_end * S0_circle

        if P0 is None:
            P0_cell = S0_cell * np.sqrt(A0_cell)
        else:
            P0_cell = P0

        total_area_target = self.num_cells * A0_cell
        R_target = np.sqrt(total_area_target / np.pi)

        # --- Génération initiale des points dans un grand disque ---
        n_generated = int(self.num_cells * generator_factor)
        R_gen = R_target * 1.5

        inner_points = []
        attempts = 0
        while len(inner_points) < n_generated and attempts < n_generated * 10:
            pt = np.random.uniform(-R_gen, R_gen, 2)
            if np.linalg.norm(pt) <= R_gen:
                inner_points.append(pt)
            attempts += 1
        inner_points = np.array(inner_points[:n_generated])
        n_generated = len(inner_points)

        if self.cut:
            mask = (np.abs(inner_points[:, 0]) > 0.1) | (inner_points[:, 1] > 0.1)
            inner_points = inner_points[mask]
            n_generated = len(inner_points)

        # --- Points fantômes très éloignés (fixes) ---
        n_boundary = max(200, n_generated // 2)
        theta = np.linspace(0, 2 * np.pi, n_boundary, endpoint=False)
        R_ghost = R_gen * 2.5
        boundary_points = np.column_stack([R_ghost * np.cos(theta),
                                        R_ghost * np.sin(theta)])

        # --- Relaxation de Lloyd ---
        for step in range(relaxation_steps):
            # Diagramme temporaire (intérieur + fantômes)
            all_pts = np.vstack([inner_points, boundary_points])
            vor = Voronoi(all_pts)

            # Calcul des centroïdes pour chaque point intérieur
            new_inner = np.zeros_like(inner_points)
            for i, idx_inner in enumerate(range(len(inner_points))):
                region = vor.regions[vor.point_region[idx_inner]]
                if -1 in region:
                    # région non bornée (ne devrait pas arriver)
                    new_inner[i] = inner_points[i]
                else:
                    vertices = vor.vertices[region]
                    new_inner[i] = vertices.mean(axis=0)
            inner_points = new_inner
            # On ne modifie pas les points fantômes

        # --- Construction définitive du Voronoï ---
        all_points = np.vstack([inner_points, boundary_points])
        vor = Voronoi(all_points)

        # --- Sommets ---
        self.vertices = {}
        vertex_map = {}
        for v_idx, pos in enumerate(vor.vertices):
            v = Vertex(id=v_idx + 1, position=pos)
            self.vertices[v.id] = v
            vertex_map[v_idx] = v

        # --- Arêtes et cellules temporaires ---
        self.edges = {}
        temp_cells = {}
        inner_indices = list(range(len(inner_points)))
        ridge_owner = {}

        def _add_edge(vert_ids):
            sorted_ids = tuple(sorted(vert_ids))
            for edge in self.edges.values():
                if tuple(sorted(edge.vertex_ids)) == sorted_ids:
                    return edge
            edge_id = len(self.edges) + 1
            edge = Edge(id=edge_id, vertex_ids=vert_ids)
            self.edges[edge_id] = edge
            return edge

        for (p1, p2), (v1, v2) in zip(vor.ridge_points, vor.ridge_vertices):
            if v1 == -1 or v2 == -1:
                continue
            owner1 = p1 if p1 in inner_indices else None
            owner2 = p2 if p2 in inner_indices else None
            cell_id1 = (inner_indices.index(owner1) + 1) if owner1 is not None else None
            cell_id2 = (inner_indices.index(owner2) + 1) if owner2 is not None else None
            vert_ids = [vertex_map[v1].id, vertex_map[v2].id]
            edge = _add_edge(vert_ids)
            if cell_id1 is not None:
                edge.add_cell_id(cell_id1)
            if cell_id2 is not None:
                edge.add_cell_id(cell_id2)
            ridge_owner[tuple(sorted(vert_ids))] = (cell_id1, cell_id2)

        # Construire les cellules
        for idx_inner in inner_indices:
            cell_id = inner_indices.index(idx_inner) + 1
            region = vor.regions[vor.point_region[idx_inner]]
            if -1 in region:
                continue
            cell_vertices = [vertex_map[v_idx] for v_idx in region]

            num_neighbors = 0
            for (v1, v2), (c1, c2) in ridge_owner.items():
                if c1 == cell_id and c2 is not None:
                    num_neighbors += 1
                elif c2 == cell_id and c1 is not None:
                    num_neighbors += 1

            temp_cells[cell_id] = {
                'vertices': cell_vertices,
                'num_neighbors': num_neighbors,
                'min_dist': self.min_distance_to_origin(cell_vertices)
            }

        # --- Sélection des cellules qui intersectent le disque de rayon R_sel ---
        sorted_cells = sorted(temp_cells.items(), key=lambda item: item[1]['min_dist'])

        if len(sorted_cells) < self.num_cells:
            raise RuntimeError(
                f"Pas assez de cellules générées ({len(sorted_cells)}) pour "
                f"atteindre num_cells = {self.num_cells}. "
                f"Augmentez generator_factor (actuel = {generator_factor})."
            )

        R_sel = sorted_cells[self.num_cells - 1][1]['min_dist']
        kept = [item for item in sorted_cells if item[1]['min_dist'] <= R_sel]

        if len(kept) > self.num_cells:
            border = [item for item in kept if item[1]['min_dist'] == R_sel]
            rng = np.random.default_rng(seed)
            to_remove = rng.choice(border, size=len(kept) - self.num_cells, replace=False)
            kept = [item for item in kept if item not in to_remove]

        kept_ids = [cell_id for cell_id, _ in kept]

        # --- Création des Cell finales ---
        self.cells = {}
        for cell_id in kept_ids:
            info = temp_cells[cell_id]
            dist_centroid = np.linalg.norm(
                np.mean([v.position for v in info['vertices']], axis=0)
            )
            if self.grad_mode in ("center", "dome", "cone"):
                relative_position = 1 - min(1, dist_centroid / R_sel)
            else:
                relative_position = 1 - (dist_centroid / R_sel) if R_sel > 0 else 0.0

            cell = Cell(
                id=cell_id,
                vertices=info['vertices'],
                num_neigh=info['num_neighbors'],
                relative_position=relative_position,
                L0=side_length,
                alpha=self.alpha,
                beta=self.beta,
                gamma=self.gamma,
                S0=S0_cell,
                A0=A0_cell,
                P0=P0_cell,
                mode=self.mode
            )
            self.cells[cell_id] = cell
            for v in info['vertices']:
                v.add_cell_id(cell_id)

        # --- Ménage ---
        edges_to_remove = [eid for eid, edge in self.edges.items()
                        if not any(cid in self.cells for cid in edge.cell_ids)]
        for eid in edges_to_remove:
            del self.edges[eid]

        vertices_to_remove = [vid for vid, v in self.vertices.items() if not v.cell_ids]
        for vid in vertices_to_remove:
            del self.vertices[vid]

        # --- Mise à l'échelle ---
        scale = R_target / R_sel
        for v in self.vertices.values():
            v.position *= scale

        # --- Finalisation ---
        self.radius = R_target
        self.update_cell_AP()
        self.update_edge_length()
        self.compute_energy()
        self.update_cell_SL(self.S_end, self.L0_end)
        self.get_boundary_vertex_ids()

        if self.cut:
            tol = side_length / 1.5
            self.right_side_vertex_ids = []
            for vid, v in self.vertices.items():
                x, y = v.position
                if 0 < x < tol and y <= tol and y >= -self.radius - tol:
                    self.right_side_vertex_ids.append(vid)


        if clip_to_circle:
            min_area = 0.05 * A0_cell   # 5% de l'aire d'une cellule cible
            self.clip_boundary_cells_to_circle(self.radius, min_area=min_area)          

        self.rotate_vertices()



    ## tol la tolérance de dépassement du cercle, min_area: les cellules don't l'aire est plus petite que ça seront tuées (0.0 => aucune tuée)
    def clip_boundary_cells_to_circle(self, radius, tol=1e-9, min_area=0.0):
        """
        Découpe les cellules dépassant du cercle en gardant la partie intérieure.
        Réordonne les sommets, nettoie les IDs de cellules orphelins dans les sommets,
        puis supprime les cellules trop petites via kill_cells.
        """
        def get_or_create_vertex(pos):
            for v in self.vertices.values():
                if np.allclose(v.position, pos, atol=1e-9):
                    return v
            new_id = max(self.vertices.keys(), default=0) + 1
            v = Vertex(id=new_id, position=pos)
            self.vertices[new_id] = v
            return v

        new_cells = {}
        for cid, cell in self.cells.items():
            verts = [self.vertices[vid] for vid in cell.vertex_ids]
            n = len(verts)
            inside = [np.linalg.norm(v.position) <= radius + tol for v in verts]

            if all(inside):
                new_cells[cid] = cell
                continue

            # Reconstruction du contour intérieur
            new_positions = []
            for i in range(n):
                A_pos = verts[i].position
                B_pos = verts[(i + 1) % n].position
                A_in = inside[i]
                B_in = inside[(i + 1) % n]

                if A_in:
                    if not new_positions or not np.allclose(new_positions[-1], A_pos, atol=1e-9):
                        new_positions.append(A_pos)
                if A_in != B_in:
                    pts = self._intersect_segment_circle(A_pos, B_pos, radius)
                    if pts:
                        pt = pts[0]
                        if not new_positions or not np.allclose(new_positions[-1], pt, atol=1e-9):
                            new_positions.append(pt)

            # Fermeture éventuelle
            if len(new_positions) >= 2 and np.allclose(new_positions[0], new_positions[-1], atol=1e-9):
                new_positions.pop()
            if len(new_positions) < 3:
                # La cellule découpée disparaît, on ne la garde pas
                continue

            new_vert_list = [get_or_create_vertex(pos) for pos in new_positions]
            cell.vertex_ids = [v.id for v in new_vert_list]
            cell.vertex_ids = cell.anticlockwise(cell.vertex_ids, self.vertices)
            for v in new_vert_list:
                if cid not in v.cell_ids:
                    v.add_cell_id(cid)
            new_cells[cid] = cell

        self.cells = new_cells

        # ==========  NETTOYAGE DES CELL_IDS ORPHELINS  ==========
        # On retire de chaque sommet tout cell_id qui n'existe plus dans self.cells
        for v in self.vertices.values():
            v.cell_ids = [cid for cid in v.cell_ids if cid in self.cells]

        # ==========  NETTOYAGE DES SOMMETS INUTILISÉS  ==========
        used_vids = set()
        for cell in self.cells.values():
            used_vids.update(cell.vertex_ids)
        for vid in list(self.vertices):
            if vid not in used_vids:
                del self.vertices[vid]

        # Reconstruction des arêtes
        self.edges = {}
        def _add_edge(vs):
            key = tuple(sorted(vs))
            for e in self.edges.values():
                if tuple(sorted(e.vertex_ids)) == key:
                    return e
            eid = len(self.edges) + 1
            edge = Edge(id=eid, vertex_ids=vs)
            self.edges[eid] = edge
            return edge

        for cell in self.cells.values():
            vids = cell.vertex_ids
            n = len(vids)
            for i in range(n):
                edge = _add_edge((vids[i], vids[(i + 1) % n]))
                if cell.id not in edge.cell_ids:
                    edge.add_cell_id(cell.id)

        self.update_cell_AP()
        self.update_edge_length()

        # Suppression des cellules trop petites (avec mise à jour de num_cells)
        if min_area > 0.0:
            to_kill = [cid for cid, cell in self.cells.items() if cell.A < min_area]
            if to_kill:
                self.kill_cells(to_kill)          # utilise votre méthode éprouvée
                self.num_cells = len(self.cells)  # synchronisation indispensable
                self.update_cell_AP()
                self.update_edge_length()




    @staticmethod
    def _intersect_segment_circle(A, B, radius):
        """
        A, B : np.array positions des extrémités
        radius : rayon du cercle centré en (0,0)
        Retourne le(s) point(s) d'intersection du segment [AB] avec le cercle,
        dans l'ordre des paramètres t croissants (0 ≤ t ≤ 1).
        """
        # Vecteur direction
        d = B - A
        # Résoudre |A + t*d|^2 = R^2
        a = np.dot(d, d)
        b = 2 * np.dot(A, d)
        c = np.dot(A, A) - radius**2
        disc = b*b - 4*a*c
        if disc < 0:
            return []
        sqrt_disc = np.sqrt(disc)
        t1 = (-b - sqrt_disc) / (2*a)
        t2 = (-b + sqrt_disc) / (2*a)
        pts = []
        for t in (t1, t2):
            if 0 <= t <= 1:
                pts.append(A + t * d)
        return pts




    @staticmethod
    def _point_to_segment_distance(px, py, ax, ay, bx, by):
        """Distance minimale d'un point P à un segment AB."""
        abx, aby = bx - ax, by - ay
        apx, apy = px - ax, py - ay
        # produit scalaire (AP·AB) / (AB·AB)
        dot = apx * abx + apy * aby
        norm2 = abx * abx + aby * aby
        t = dot / norm2
        t = max(0.0, min(1.0, t))
        near_x = ax + t * abx
        near_y = ay + t * aby
        return np.hypot(px - near_x, py - near_y)

    @staticmethod
    def min_distance_to_origin(vertices):
        """Distance minimale entre l'origine (0,0) et le polygone formé par les vertices."""
        pts = np.array([v.position for v in vertices])
        n = len(pts)

        # Test rapide : si l'origine est à l'intérieur du polygone, distance = 0
        # On utilise le test du nombre d'intersections d'une demi‑droite horizontale.
        inside = False
        x, y = 0.0, 0.0
        for i in range(n):
            x1, y1 = pts[i]
            x2, y2 = pts[(i + 1) % n]
            # vérifier si le rayon horizontal depuis (x,y) coupe le segment
            if ((y1 > y) != (y2 > y)) and (x < (x2 - x1) * (y - y1) / (y2 - y1) + x1):
                inside = not inside
        if inside:
            return 0.0

        # Sinon, distance minimale aux arêtes
        min_d = float('inf')
        for i in range(n):
            x1, y1 = pts[i]
            x2, y2 = pts[(i + 1) % n]
            d = Mesh._point_to_segment_distance(0.0, 0.0, x1, y1, x2, y2)
            if d < min_d:
                min_d = d
        return min_d


    @staticmethod
    def _regular_shape_factor(n_sides):
        """Facteur de forme d'un polygone régulier à n_sides côtés."""
        if n_sides < 3:
            return 2 * np.sqrt(np.pi)   # cas circulaire
        return np.sqrt(4 * n_sides * np.tan(np.pi / n_sides))


    def _add_edge(self, vertex_ids):
        """Ajoute une arête (ou la retourne si elle existe déjà)."""
        sorted_ids = tuple(sorted(vertex_ids))
        # cherche si une arête avec ces mêmes vertex ids existe déjà
        for edge in self.edges.values():
            if tuple(sorted(edge.vertex_ids)) == sorted_ids:
                return edge
        # sinon création
        edge_id = len(self.edges) + 1
        edge = Edge(id=edge_id, vertex_ids=vertex_ids)
        self.edges[edge_id] = edge
        return edge
    

































