# Python Project Setup Guide

This repository contains a Python project that requires specific dependencies to run successfully. Below are the details of the required packages and the recommended Python version.

## Recommended Python Version

To ensure compatibility, it is recommended to use **Python 3.9 or later**.

## Required Packages

The following Python packages are necessary to run the project:

- `numpy`: For numerical computations.
- `shapely`: Provides utilities for geometric operations.
- `matplotlib`: Used for creating static, animated, and interactive visualizations.
- `os`: Provides functions for interacting with the operating system.
- `scipy`: Includes modules for optimization, integration, and scientific computations.
- `mpl_toolkits.mplot3d`: Enables 3D plotting in Matplotlib.
- `pandas`: Provides data manipulation and analysis tools.

### Specific Imports

Below is a breakdown of the imports used in the project:

```python
import numpy as np 
from shapely.geometry import Polygon  
import matplotlib.pyplot as plt  
import os  
from scipy.optimize import minimize 
from mpl_toolkits.mplot3d.art3d import Poly3DCollection  
from scipy.spatial import Delaunay  
from matplotlib.colors import Normalize 
import matplotlib.cm as cm  
import pandas as pd  



To install these packages, run the following command:

```bash
pip install numpy shapely matplotlib scipy pandas