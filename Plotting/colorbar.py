import matplotlib as mpl
import numpy as np

from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap, SymLogNorm
from scipy.interpolate import CubicSpline


def make_colormap(gamma_pre: float, gamma_post: float) -> list[ListedColormap, SymLogNorm]:
    
    Ncolors = 256 
    base = np.arange(Ncolors) / Ncolors
    
    nodes = np.linspace(0.0, 1.0, 4)
    weights_blue = [0, 0.33, 0.87, 1.0] # weights set by Shankar, recommended not to change
    weights_red = [0, 0.24, 0.45, 1.0]
    
    def make_channel(weights):
        return CubicSpline(nodes, weights, bc_type=("natural", "clamped"))(base ** gamma_pre)
    
    blue_channel = make_channel(weights_blue) ** gamma_post
    red_channel = green_channel = make_channel(weights_red) ** gamma_post
    alpha = np.ones_like(base)
    
    RGBA = np.array([red_channel, green_channel, blue_channel, alpha]).T
    colormap = ListedColormap(RGBA).reversed()
    
    norm = SymLogNorm(1e-3, linscale = 0.2, vmin = 0, vmax = 0.25)
     
    return colormap, norm
