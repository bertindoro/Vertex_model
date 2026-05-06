import numpy as np



class Vertex:
    def __init__(self, id, position):
        """
        Initialize a vertex 
        
        Parameters:
        - id: unique id of the vertex
        - position: coordinates of the vertex 
        """
        self.id = id
        self.cell_ids = []
        self.position = np.array(position, dtype=np.float64)

    def add_cell_id(self, cell_id):
        """
        Add a cell id to the list of cells associated with this vertex if not already present
        """
        if cell_id not in self.cell_ids:
            self.cell_ids.append(cell_id)


class Edge:
    def __init__(self, id, vertex_ids):
        """
        Initialize an edge
        
        Parameters:
        - id: unique id of the edge
        - vertex_ids: tuple of two vertex ids defining this edge
        """
        self.id = id
        self.vertex_ids = tuple(sorted(vertex_ids))  # Ensure consistency in vertex order
        self.cell_ids = []  # Cells sharing this edge
        self.L = 0.0  # Initialize length

    def add_cell_id(self, cell_id):
        """
        Add a cell id that shares this edge
        """
        if cell_id not in self.cell_ids:
            self.cell_ids.append(cell_id)


    def remove_cell_id(self, cell_id):
        """
        Remove a cell id from the list of cells sharing this edge
        """
        if cell_id in self.cell_ids:
            self.cell_ids.remove(cell_id)


    def length(self, vertices):
        """
        Calculate and update the length of the edge based on vertex positions
        
        Parameters:
        - vertices: Dictionary of vertex indexed by their ids
        """
        v1 = vertices[self.vertex_ids[0]].position
        v2 = vertices[self.vertex_ids[1]].position
        self.L = np.linalg.norm(np.array(v1) - np.array(v2))


class Cell:
    def __init__(self, id, vertices, num_neigh, relative_position = 1, L0=1, alpha = 1, beta = 1, gamma = 0, P0 = None, A0 = None, S0 = 2*np.sqrt(np.pi), mode = "hexagon"):
        """
        Initialize a cell 
        
        Parameters:
        - id: unique id of the cell
        - vertices: list of vertex defining the cell 
        - num_neigh: number of neighboring cells
        - relative_position: relative position parameter used in growth gradients (depends on the gradient type)
        - L0: target length scale for the cell 
        - alpha, beta, gamma: coefficients for area, perimeter, and adhesion terms
        - P0, A0: preferred perimeter and area (if None, computed based on mode and L0)
        - S0: shape parameter
        - mode: defined the preferred geometry of the cell: "circle", "triangle", "three_triangles", "hexagon"
        """
        self.id = id
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.relative_position = relative_position
        self.num_neighbors = num_neigh
        vertex_ids = [vertex.id for vertex in vertices]
        for vertex in vertices:
            vertex.add_cell_id(self.id)
        self.vertex_ids = self.anticlockwise(vertex_ids, {v.id: v for v in vertices})
        self.is_boundary = False
        self.S0 = S0
        if A0 is None:
            if mode == "circle":
                self.A0 = np.pi *L0**2    # circle
            elif mode == "triangle":
                self.A0 = (L0/2)**2 * np.sqrt(3)  #triangle
            elif mode == "three_triangles":
                self.A0 = (L0/2)*(L0/2)*np.tan(np.pi/6)
            else:
                self.A0 = (3*np.sqrt(3)/2)*L0**2   # hexagon
        else:
            self.A0 = A0
        if P0 is None:
            self.P0 = S0 * np.sqrt(self.A0)
        else:
            self.P0 = P0
        self.A = None
        self.P = None


    def update_AP(self, vertices):
        """
        Update the cell's current area and perimeter based on vertex positions
        
        Parameters:
        - vertices: dictionary of vertex indexed by their ids
        """
        first_key = self.vertex_ids[0]
        if len(vertices[first_key].position)==3:
            self.A = self.area_3d(vertices)
        else:
            self.A = self.area(vertices)
        self.P = self.perimeter(vertices)


    def update_A(self, vertices):
        """
        Update only the cell's area
        
        Parameters:
        - vertices: dictionary of vertex indexed by their ids
        """
        if len(vertices[1].position)==3:
            self.A = self.area_3d(vertices)
        else:
            self.A = self.area(vertices)


    def update_SL(self, new_S, cst, new_L0, mode, grad_S, grad_L0):
        """
        Update preferred area and perimeter based on gradients or growth
        
        Parameters:
        - new_S: new shape parameter 
        - cst: isoperimetric value depending on the cell type
        - new_L0: new target size length
        - mode: preferred shape of the cell ("circle", "triangle", "hexagon", "three_triangles")
        - grad_S: Boolean indicating if perimeter scaling uses a gradient.
        - grad_L0: Boolean indicating if length scaling uses a gradient.

        Remark: the gradient of L0 can be changed for an inverse gradient
        """
        if grad_S:
            new_S = 1-(1- self.relative_position)*(1-new_S)
        if grad_L0:
            new_L0 = (1+ (1-self.relative_position)*(new_L0 - 1))
            #new_L0 = (1+ (self.relative_position)*(new_L0 - 1))        #inverse gradient: more growth inside than on the boundary
        new_S0 = new_S*cst
        if mode == "circle":
            self.A0 = np.pi *new_L0**2
        elif mode == "triangle":
            self.A0 = (new_L0/2)**2 * np.sqrt(3)
        elif mode == "three_triangles":
            self.A0 = (new_L0/2)*(new_L0/2)*np.tan(np.pi/6)
        else: 
            self.A0 = (3*np.sqrt(3)/2)*new_L0**2 
        self.P0 = new_S0 * np.sqrt(self.A0)


    def anticlockwise(self, vertex_ids, vertices_dict):
        """
        Order vertex ids in anticlockwise order 
        
        Parameters:
        - vertex_ids: list of vertex ids (vertices in the cell)
        - vertices_dict: dictionary of vertex indexed by their ids
        
        Returns:
        - sorted list of vertex ids in anticlockwise order
        """
        positions = np.array([vertices_dict[v_id].position for v_id in vertex_ids])
        center = np.mean(positions, axis=0)
        angles = np.arctan2(positions[:, 1] - center[1], positions[:, 0] - center[0])
        sorted_indices = np.argsort(angles)
        return [vertex_ids[i] for i in sorted_indices]


    def area(self, vertices):
        """
        calculate polygon area in 2D using shoelace formula
        
        Parameters:
        - vertices: dictionary of vertex indexed by their ids
        
        Returns:
        - area as float
        """
        positions = np.array([vertices[v_id].position for v_id in self.vertex_ids])
        n = len(positions)
        area = 0.0
        for i in range(n):
            j = (i + 1) % n
            area += positions[i][0] * positions[j][1] - positions[j][0] * positions[i][1]
        return 0.5 * np.abs(area)


    def area_3d(self, vertices):
        """
        compute area of polygon in 3D by decomposing into triangles around centroid
        
        Parameters:
        - vertices: dictionary of vertex indexed by their ids
        
        Returns:
        - area as float
        """
        positions = np.array([vertices[v_id].position for v_id in self.vertex_ids])
        n = len(positions)
        if n <= 2:
            raise ValueError("This is a line or a point")
        
        center = np.mean(positions, axis=0)

        total_area = 0.0

        for i in range(n):
            current_vertex = positions[i]
            next_vertex = positions[(i + 1) % n]
            
            # Define triangle
            edge1 = current_vertex - center
            edge2 = next_vertex - center
            cross_product = np.cross(edge1, edge2)
            
            # The area of the triangle is half the magnitude of the cross product
            triangle_area = 0.5 * np.linalg.norm(cross_product)
            total_area += triangle_area
        
        return total_area


    def perimeter(self, vertices):
        """
        Calculate perimeter of the polygon
        
        Parameters:
        - vertices: dictionary of vertex indexed by their ids
        
        Returns:
        - perimeter as float
        """
        positions = np.array([vertices[v_id].position for v_id in self.vertex_ids])
        n = len(positions)
        if n < 2:
            return 0.0  
        perimeter = 0.0
        for i in range(n):
            j = (i + 1) % n
            perimeter += np.linalg.norm(positions[i] - positions[j])  
        return perimeter


    def gradient_area(self, vertex_id, vertices):
        """
        Calculate gradient of the area with respect to the position of the given vertex
        
        Parameters:
        - vertex_id: id of the vertex to calculate gradient for
        - vertices: dictionary of vertex indexed by their ids
        
        Returns:
        - area gradient vector with respect to vertex position
        """
        positions = np.array([vertices[v_id].position for v_id in self.vertex_ids])
        n = len(positions)
        cst = (self.alpha/2)*(1- self.A0/self.A)

        v_index = self.vertex_ids.index(vertex_id)
        prev_idx = (v_index - 1) % n
        next_idx = (v_index + 1) % n

        grad_x = (positions[next_idx][1] - positions[prev_idx][1])*cst
        grad_y = (positions[prev_idx][0] - positions[next_idx][0])*cst

        temp = 0

        for i in range(n):
            next_idx = (i + 1) % n  
            temp += positions[i][0] * positions[next_idx][1] - positions[next_idx][0] * positions[i][1]

        return np.array([grad_x*temp, grad_y*temp])
    

    def gradient_perimeter(self,vertex_id, vertices):
        """
        Calculate gradient of the perimeter with respect to the position of the given vertex
        
        Parameters:
        - vertex_id: id of the vertex to calculate gradient for
        - vertices: dictionary of vertex indexed by their ids
        
        Returns:
        - perimeter gradient vector of with respect to vertex position
        """
        positions = np.array([vertices[v_id].position for v_id in self.vertex_ids])
        n = len(positions)

        sum_inv_perim = np.sum(1 / np.linalg.norm(positions[i] - positions[(i+1) % n]) for i in range(n))
        cst = 2*self.beta*len(self.vertex_ids)*(self.P - self.P0)/sum_inv_perim
        
        v_index = self.vertex_ids.index(vertex_id)
        prev_idx = (v_index - 1) % n
        next_idx = (v_index + 1) % n

        grad_x = 2*positions[v_index][0] - positions[prev_idx][0] - positions[next_idx][0]
        grad_y = 2*positions[v_index][1] - positions[prev_idx][1] - positions[next_idx][1]


        return np.array([grad_x*cst, grad_y*cst])

 
    def gradient_adhesion(self, vertex_id, vertices):
        """
        Calculate gradient of adhesion energy with respect to the position of the given vertex
        
        Parameters:
        - vertex_id: id of the vertex to calculate gradient for
        - vertices: dictionary of vertex indexed by their ids
        
        Returns:
        -adhesion gradient vector 
        """
        positions = np.array([vertices[v_id].position for v_id in self.vertex_ids])
        n = len(positions)
        v_index = self.vertex_ids.index(vertex_id)
        prev_idx = (v_index - 1) % n
        next_idx = (v_index + 1) % n
        dki_1 = np.sqrt((positions[prev_idx][0] - positions[v_index][0])**2 + (positions[prev_idx][1] - positions[v_index][1])**2  )
        dki = np.sqrt((positions[v_index][0] - positions[next_idx][0])**2 + (positions[v_index][1] - positions[next_idx][1])**2 )
        grad_x = -(self.gamma/dki_1)*(positions[prev_idx][0] - positions[v_index][0]) + (self.gamma/dki)*(positions[v_index][0] - positions[next_idx][0])
        grad_y = -(self.gamma/dki_1)*(positions[prev_idx][1] - positions[v_index][1]) + (self.gamma/dki)*(positions[v_index][1] - positions[next_idx][1])
        return np.array([grad_x, grad_y])


    def energy_area(self):
        """
        Compute area energy term
        """
        result = self.alpha * (self.A - self.A0) ** 2
        return result if abs(result) > 1e-15 else 0.0


    def energy_perimeter(self):
        """
        Compute perimeter energy term
        """
        result = self.beta * (self.P - self.P0) ** 2
        return result if abs(result) > 1e-15 else 0.0


    def energy_adhesion(self):
        """
        Compute adhesion energy term
        """
        result = self.gamma * self.P
        return result if abs(result) > 1e-15 else 0.0   



    def center_of_cell(self, vertices):
        """
        Compute the position of the center of the cell using the centroid method.
        
        Args:
            vertices: List of vertex objects with position attribute
            
        Returns:
            (cx, cy): Coordinates of the cell center
        """
        positions = np.array([vertices[v_id].position for v_id in self.vertex_ids])
        n = len(positions)
        
        if n < 3:
            raise ValueError("A cell must have at least 3 vertices")
        
        # Initialize variables
        area = 0.0
        cx = 0.0
        cy = 0.0
        
        # Calculate area and centroid coordinates
        for i in range(n):
            j = (i + 1) % n
            cross = positions[i][0] * positions[j][1] - positions[j][0] * positions[i][1]
            area += cross
            cx += (positions[i][0] + positions[j][0]) * cross
            cy += (positions[i][1] + positions[j][1]) * cross
        
        # Finalize calculations
        area *= 0.5
        
        if area == 0:
            # Degenerate polygon, return average of vertices
            return np.mean(positions, axis=0)
        
        cx /= (6 * area)
        cy /= (6 * area)
        
        return np.array([cx, cy])
    


    def update_neighbors_and_boundary(self, edges):
        """
        Update the value of num_neighbors and is_boundary based on the edges

        Args:
            edges : dict {edge_id: Edge}
        """
        neighbors = set()
        on_boundary = False

        for edge in edges.values():
            if self.id not in edge.cell_ids:
                continue

            # Quelles sont les autres cellules sur cette arête ?
            other_cells = [cid for cid in edge.cell_ids if cid != self.id]

            if len(other_cells) == 1:
                # Arête interne → l'autre cellule est une voisine
                neighbors.add(other_cells[0])
            elif len(other_cells) == 0:
                # Arête qui n'appartient qu'à cette cellule → frontière
                on_boundary = True

        self.num_neighbors = len(neighbors)
        self.is_boundary = on_boundary