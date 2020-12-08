"""
dx1/dt = x1*(alpha-beta*x2)
dx2/dt = -x2*(gamma-delta*x1)
"""

import numpy as np
import matplotlib.pyplot as plt


# définition des constantes
alpha, beta, gamma, delta = 2/3, 4/3, 1, 1

def f(x1, x2):
    return np.array([x1*(alpha - beta*x2), -x2*(gamma - delta*x1)])

def affiche_champ_de_vecteur(limite, couleur = "orange"):
    x = np.linspace(0,limite,limite + 1)
    y = np.linspace(0, limite,limite + 1)


    X, Y = np.meshgrid(x, y)
    Z = f(X, Y)

    plt.quiver(X, Y, Z[0], Z[1], color = couleur)
    plt.show()
