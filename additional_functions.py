import numpy as np
import matplotlib.pyplot as plt
from mesh import *
from scipy.optimize import curve_fit
import os
import pandas as pd

def S0_with_dome_hexagon(num_cells=2000, folder_name="S0_dome_hexagon", side_length=1):
    """
    Generates a hemispherical hexagonal mesh, computes S and S0, and plots their relationships with angular displacement

    Parameters:
    - num_cells: number of cells (default: 2000)
    - folder_name: name of the folder to save results (default="S0_dome_hexagon")
    - side_length: side length for hexagonal cells (default=1)
    """
    num_cells = num_cells
    grad_mode = "dome"
    mode = "hexagon"
    cut = False
    theta_list = []
    S_list = []
    S0_list = []

        
    def angle_between_points(A, B, C):
        """
        Calculate the angle between three points in 3D space.
        
        Parameters:
        - A, B, C: Arrays or lists representing points in 3D space.
        
        Returns:
        - Angle in degrees between the vectors AB and BC, constrained between 0° and 90°.
        """
        # Convert points to NumPy arrays
        A, B, C = map(np.array, (A, B, C))
            
        # Calculate vectors AB and BC
        AB = B - A
        BC = C - B
            
        # Compute the dot product and magnitudes of the vectors
        dot_product = np.dot(AB, BC)
        magnitude_AB = np.linalg.norm(AB)
        magnitude_BC = np.linalg.norm(BC)
            
        # Calculate the cosine of the angle
        cos_theta = dot_product / (magnitude_AB * magnitude_BC)
        
        # Clip the value to avoid any potential numerical issues with arccos
        cos_theta = np.clip(cos_theta, -1.0, 1.0)
            
        # Calculate the angle in radians
        angle_rad = np.arccos(cos_theta)
            
        # Convert the angle to degrees
        angle_deg = np.degrees(angle_rad)
        
        # If the angle is greater than 90°, subtract it from 180° to ensure it's less than 90°
        if angle_deg > 90:
            angle_deg = 180 - angle_deg
            
        return angle_deg


    M1=Mesh(num_cells, mode = mode, cut = cut, grad_mode=grad_mode, side_length=side_length)
    rho = M1.radius *np.sqrt(0.5)       # the exact value for half a sphere
    h=(M1.radius**2)/(2*rho)
    gb=np.sqrt(M1.radius**2 - h**2)
    M2=M1.copy_scale_mesh(scale_factor=(gb/M1.radius))

    M2.get_z_coordinate_dome(rho=rho, h=h, mesh=M1)
    M2.update_cell_AP() 
    M2.plot_3d_incompatibility_dome(rho=rho, h=h)
    global_S0 = M2.global_S0(gb)
    M1.get_AP_from_other_mesh(M2)
    M1.radius=gb

    #Define my three points for the angle
    # p2 center of the sphere, p1 top position of the sphere
    p2 = [0.0, 0.0, h-rho]
    p1 = [0.0, 0.0, 2*rho]

    for cell in M2.cells.values():
        positions = np.array([M2.vertices[vid].position for vid in cell.vertex_ids])
        x = np.mean(positions[:, 0])  # Mean of x-coordinates
        y = np.mean(positions[:, 1])  # Mean of y-coordinates
        z = np.mean(positions[:, 2])  # Mean of z-coordinates
        S0 = cell.P / np.sqrt(cell.A)
        S = S0 / M2.cst_S0
        p3 = [x, y, z]
        theta = angle_between_points(p1, p2, p3)
        theta_list.append(theta)
        S_list.append(S)
        S0_list.append(S0)

    
        # Save theta, a, and b to a file
        script_dir = os.path.dirname(os.path.abspath(__file__))
        output_folder = os.path.join(script_dir, folder_name)
        os.makedirs(output_folder, exist_ok=True)  # Ensure the folder exists

        # Path to the fit_constants.txt file
        curve_theta_S0_file = os.path.join(output_folder, f"curve_theta_S0_{global_S0}.txt")

        # Save the constants
        with open(curve_theta_S0_file, 'a') as file:
            file.write(f"theta: {theta}, S0: {S0}\n")

        # Data frame 
        curve_theta_S0_file_df = os.path.join(output_folder, f"curve_theta_S0_{global_S0}_df.csv")
        # Prepare the data for saving
        data = {
                    "theta": [theta],
                    "S0": [S0]
                }

        # Convert to DataFrame
        df = pd.DataFrame(data)

        # Write to CSV (append if the file exists)
        if os.path.exists(curve_theta_S0_file):
            df.to_csv(curve_theta_S0_file_df, mode='a', header=False, index=False)  # Append without header
        else:
            df.to_csv(curve_theta_S0_file_df, mode='w', header=True, index=False)   # Write with header        


    

    # Create plots with fitted curves
    fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharey=False)

    # Scatter plot for S vs theta
    axes[0].scatter(theta_list, S_list, color='blue', alpha=0.7, label='S')
    axes[0].axhline(y=1, color='red', linestyle='--', label='S = 1')
    axes[0].set_title('S vs theta')
    axes[0].set_xlabel('theta')
    axes[0].set_ylabel('S')
    axes[0].grid(True)
    axes[0].legend()

    # Dynamically set the y-axis for the first plot
    axes[0].set_ylim(min(S_list) - 0.1, max(S_list) + 0.1)

    # Scatter plot for S0 vs theta
    axes[1].scatter(theta_list, S0_list, color='green', alpha=0.7, label='S0')
    axes[1].axhline(y=M2.cst_S0, color='orange', linestyle='--', label=f'S0 = {M2.cst_S0}')
    axes[1].set_title('S0 vs theta')
    axes[1].set_xlabel('theta')
    axes[1].grid(True)
    axes[1].legend()

    # Dynamically set the y-axis for the second plot
    axes[1].set_ylim(min(S0_list) - 0.1, max(S0_list) + 0.1)

    # Set x-axis limits for consistency (optional)
    axes[0].set_xlim(min(theta_list) - 1, max(theta_list) + 1)
    axes[1].set_xlim(min(theta_list) - 1, max(theta_list) + 1)

    plt.tight_layout()
    plt.show()




def gradual_fit_dome(rho_cst_list=1, num_cells=500, folder_name="Spherical_cap", cut = True, side_length=1, mode="triangle"):
    """
    Perform the flattening of a spherical cap. Given a list of rho values, it increases the size of the spherical cap gradually

    Parameters:
    - rho_cst_list (float, int, or list): controls the size of the spherical cap
    - num_cells: number of cells 
    - folder_name: folder name to save results
    - cut: whether the mesh is cut 
    - side_length: side length 
    - mode: geometry of the cells ("triangle", "hexagon", "circle")
    """
    if isinstance(rho_cst_list, float) or isinstance(rho_cst_list, int):
        rho_cst_list = [rho_cst_list]

    rho_cst_list = np.array(rho_cst_list)
    rho_cst_list = np.sort(rho_cst_list)[::-1]

    num_cells = num_cells
    grad_mode = "dome" 

    alpha_list = [1]
    beta_list = [1]


    def model_func(x, a, b):
        return a * np.power(x, b)

    
    def angle_between_points(A, B, C):
        """
        Calculate the angle between three points in 3D space.
        
        Args:
            A, B, C: Arrays or lists representing points in 3D space.
        
        Returns:
            Angle in degrees between the vectors AB and BC, constrained between 0° and 90°.
        """
        # Convert points to NumPy arrays
        A, B, C = map(np.array, (A, B, C))
            
        # Calculate vectors AB and BC
        AB = B - A
        BC = C - B
            
        # Compute the dot product and magnitudes of the vectors
        dot_product = np.dot(AB, BC)
        magnitude_AB = np.linalg.norm(AB)
        magnitude_BC = np.linalg.norm(BC)
            
        # Calculate the cosine of the angle
        cos_theta = dot_product / (magnitude_AB * magnitude_BC)
        
        # Clip the value to avoid any potential numerical issues with arccos
        cos_theta = np.clip(cos_theta, -1.0, 1.0)
            
        # Calculate the angle in radians
        angle_rad = np.arccos(cos_theta)
            
        # Convert the angle to degrees
        angle_deg = np.degrees(angle_rad)
        
        # If the angle is greater than 90°, subtract it from 180° to ensure it's less than 90°
        if angle_deg > 90:
            angle_deg = 180 - angle_deg
            
        return angle_deg


    for alpha in alpha_list:
        
        for beta in beta_list:
            #folder_name_b = folder_name + f"_{alpha}alpha" + f"_{beta}beta"
            folder_name_b = folder_name
            M1=Mesh(num_cells, mode = mode, grad_mode=grad_mode, cut=cut, alpha=alpha, beta=beta, side_length=side_length)
            M1.save_half_simulation(folder_name=folder_name_b)
            M1_bis=M1.copy_scale_mesh(scale_factor=1)

            

            if mode=="triangle":
                #find the id of vertices on the two extremities of the half sphere
                left_id = max(M1.fix_vertex_y, key=lambda id: np.linalg.norm(M1.vertices[id].position))
                # Sort the ids by descending norm of the vertex position
                top_three_ids = sorted(M1.right_side_vertex_ids, 
                                    key=lambda id: np.linalg.norm(M1.vertices[id].position), reverse=True)[:3]
                vert1, vert2, vert3 = top_three_ids


            for rho_cst in rho_cst_list:
                rho = M1.radius *np.sqrt(0.5)*rho_cst  
                h=(M1.radius**2)/(2*rho)
                gb=np.sqrt(M1.radius**2 - h**2)
                

                M2=M1.copy_scale_mesh(scale_factor=(gb/M1.radius))
                

                M2.get_z_coordinate_dome(rho=rho, h=h, mesh=M1_bis)
                position = np.array([0, 0])
                for vertex in M1_bis.vertices.values():
                    if np.linalg.norm(vertex.position)>np.linalg.norm(position):
                        position = np.array(vertex.position)
                        id = vertex.id     
                p3_id = id
                M2.update_cell_AP()
                #M2.update_cell_A()
                global_S0 = M2.global_S0(gb)                  

                #Define my three points for the angle
                p2 = [0.0, 0.0, h-rho]
                p1 = [0.0, 0.0, 2*rho]
                p3 = M2.vertices[p3_id].position
                theta = angle_between_points(p1, p2, p3)
                M2.save_3d_dome(rho=rho, h=h, theta=theta, folder_name=folder_name_b)
                
                M1_bis.get_AP_from_other_mesh(M2)
                M1_bis.radius=gb
                
                M1_bis.equilibrium()

                if mode == "triangle":
                    #fix the two vertices for half a sphere
                    M1_bis.vertices[left_id].position[0] -= 2
                    pos1 = M1_bis.vertices[vert3].position
                    pos2 = M1_bis.vertices[vert2].position
                    vec = pos2 - pos1
                    new_pos3 = pos1 + 3 * vec
                    M1_bis.vertices[vert1].position = new_pos3 
                    M1_bis.equilibrium()

                M1_bis.save_half_flat_dome(theta=theta, folder_name=folder_name_b)



                # Extract vertices to fit
                vertices_to_fit = [M1_bis.vertices[vid].position for vid in M1_bis.right_side_vertex_ids]

                # Convert to numpy arrays
                x = np.array([v[0] for v in vertices_to_fit])
                y = np.array([v[1] for v in vertices_to_fit])


                # Filter valid indices
                valid_indices = x > 0
                x_fit = x[valid_indices]
                y_fit = y[valid_indices]

                
                x_save = x
                y_save = y
                
                initial_guess = [1, 1] 
                params, _ = curve_fit(model_func, x_fit, y_fit, p0=initial_guess)
                a, b = params


                x_curve = np.linspace(min(x_fit), max(x_fit), 1000)    
                M1_bis.save_plot_with_fit(x_curve=x_curve, a=a, b=b,global_S0=global_S0 , folder_name=folder_name_b)


                # Save theta, a, and b to a file
                script_dir = os.path.dirname(os.path.abspath(__file__))
                output_folder = os.path.join(script_dir, folder_name_b)
                os.makedirs(output_folder, exist_ok=True)  # Ensure the folder exists

                # Path to the fit_constants.txt file
                fit_constants_file = os.path.join(output_folder, "fit_constants.txt")

                # Save the constants
                with open(fit_constants_file, 'a') as file:
                    #file.write(f"alpha: {alpha}, beta: {beta}, global_S0: {global_S0}, theta: {theta}, a: {a}, b: {b}\n")
                    file.write(f"global_S0: {global_S0}, theta: {theta}, a: {a}, b: {b}\n")

                # Data frame 
                fit_constants_file_df = os.path.join(output_folder, "fit_constants_df.csv")
                # Prepare the data for saving
                data = {
                    #"alpha": [alpha],
                    #"beta": [beta],
                    "global_S0": [global_S0],  # Save as a list to append multiple rows later
                    "theta": [theta],
                    "a": [a],
                    "b": [b]
                }

                # Convert to DataFrame
                df = pd.DataFrame(data)

                if not os.path.exists(fit_constants_file_df):
                    # Explicitly create the file and write the header
                    with open(fit_constants_file_df, 'w') as f:
                        f.write(','.join(df.columns) + '\n')
                    df.to_csv(fit_constants_file_df, mode='a', header=False, index=False)
                else:
                    df.to_csv(fit_constants_file_df, mode='a', header=False, index=False)






                # Save the coordinates of points in a file
                script_dir = os.path.dirname(os.path.abspath(__file__))
                output_folder = os.path.join(script_dir, folder_name_b)
                os.makedirs(output_folder, exist_ok=True)  # Ensure the folder exists



                # Save the coordinates for every iteration
                coordinates_file_txt = os.path.join(output_folder, "coordinates.txt")
                coordinates_file_df = os.path.join(output_folder, "coordinates_df.csv")

                # Save the coordinates in a text file
                with open(coordinates_file_txt, 'a') as file:
                    file.write(f"Theta: {theta}, Global_S0: {global_S0}\n")
                    file.write("x_save and y_save coordinates:\n")
                    for x_val, y_val in zip(x_save, y_save):
                        file.write(f"{x_val}, {y_val}\n")
                    file.write("\n")  # Add a blank line for readability

                # Prepare the data for saving in the DataFrame
                data_coordinates = {
                    "theta": [theta] * len(x_save),  # Repeat theta for all coordinates
                    "global_S0": [global_S0] * len(x_save),  # Repeat global_S0 for all coordinates
                    "x_save": x_save,  # Save x_save coordinates
                    "y_save": y_save   # Save y_save coordinates
                }

                # Convert to DataFrame
                df_coordinates = pd.DataFrame(data_coordinates)

                # Save to CSV
                if not os.path.exists(coordinates_file_df):
                    # Explicitly create the file and write the header
                    with open(coordinates_file_df, 'w') as f:
                        f.write(','.join(df_coordinates.columns) + '\n')
                    df_coordinates.to_csv(coordinates_file_df, mode='a', header=False, index=False)
                else:
                    df_coordinates.to_csv(coordinates_file_df, mode='a', header=False, index=False)




def do_cone(num_cells = 500, cut = True, theta = 90, mode = "triangle", folder_name = "Cone"):
    """
    Perform the flattening for the cone

    Parameters:
        num_cells: number of cells 
        cut: whether the mesh is cut
        theta: target opening angle
        mode: mesh geometry ("triangle", "hexagon", "circle")
        folder_name: folder name for saving output files
    """
    M1=Mesh(num_cells, mode = mode, cut = cut, theta=theta)
    if mode=="triangle":
                # Sort the ids by descending norm of the vertex position
                top_three_ids = sorted(M1.right_side_vertex_ids, 
                                    key=lambda id: np.linalg.norm(M1.vertices[id].position), reverse=True)[:3]
                vert1, vert2, vert3 = top_three_ids
    gb = M1.radius*(1-theta/360)
    h = np.sqrt(M1.radius**2 - gb**2)
    M2=M1.copy_scale_mesh(scale_factor=(gb/M1.radius))
    M2.get_z_coordinate_cone(gb=gb, h=h)
    M2.update_cell_AP() 
    M2.save_3d_cone(gb=gb, h=h, folder_name=folder_name)
    M1.get_AP_from_other_mesh(M2)
    M1.radius=gb
    M1.equilibrium()
    if mode == "triangle":
                    #fix the two vertices for half a sphere
                    pos1 = M1.vertices[vert3].position
                    pos2 = M1.vertices[vert2].position
                    vec = pos2 - pos1
                    new_pos3 = pos1 + 2 * vec
                    M1.vertices[vert1].position = new_pos3 
                    M1.equilibrium()
    

    M1.save_half_flat_dome(theta=theta, folder_name=folder_name)






def is_anticlockwise(points):
    """
    Vérifie si une liste de points est dans l'ordre anti-horaire.
    
    Args:
        points: Liste de tuples (x, y) ou tableau numpy de forme (n, 2)
        
    Returns:
        True si les points sont dans l'ordre anti-horaire, False sinon
    """
    # Convertir en tableau numpy si nécessaire
    if not isinstance(points, np.ndarray):
        points = np.array(points)
    
    n = len(points)
    if n < 3:
        # Un polygone valide doit avoir au moins 3 points
        # Pour moins de 3 points, on considère que l'ordre n'est pas défini
        return False
    
    # Calcul de l'aire signée (formule du shoelace)
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += points[i][0] * points[j][1] - points[j][0] * points[i][1]
    
    # Si l'aire est positive -> ordre anti-horaire
    # Si l'aire est négative -> ordre horaire
    return area > 0



