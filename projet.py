"""
dx1/dt = x1*(alpha-beta*x2)
dx2/dt = -x2*(gamma-delta*x1)
"""
import autograd.numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from precedent_projet import courbe_niveau


# définition des constantes
alpha, beta, gamma, delta = 2/3, 4/3, 1, 1.1
eps = 10**(-1)


# définition des fonctions
def f(x1, x2):
    return np.array([x1*(alpha - beta*x2), -x2*(gamma - delta*x1)])

def H(x1, x2):
    return delta*x1 - gamma* np.log(x1) + beta*x2 - alpha*np.log(x2)


# définition des réponses aux questions
def affiche_champ_de_vecteur(limite = 10, couleur = "orange"):

    x = np.linspace(0,limite,limite + 1)
    y = np.linspace(0, limite,limite + 1)


    X, Y = np.meshgrid(x, y)
    Z = f(X, Y)

    plt.quiver(X, Y, Z[0], Z[1], color = couleur)
    plt.show()

def affiche_portrait_de_phase(limite = 10):

    x = np.linspace(0,limite,limite + 1)
    y = np.linspace(0, limite,limite + 1)

    X, Y = np.meshgrid(x, y)
    Z = f(X, Y)

    U, V = Z[0], Z[1]
    vitesse = np.sqrt(U**2 + V**2)

    
    fig = plt.figure(figsize=(7, 9))
    gs = gridspec.GridSpec(nrows=2, ncols=2, height_ratios=[1, 1])

    #  changer la densité le long d'une ligne de champ
    ax0 = fig.add_subplot(gs[0, 0])
    ax0.streamplot(X, Y, U, V, density=[0.5, 1], color = "orange")
    ax0.set_title('Variation de la densité')

    # changer la couleur le long d'une ligne
    ax1 = fig.add_subplot(gs[0, 1])
    strm = ax1.streamplot(X, Y, U, V, color=U, linewidth=2, cmap='gist_heat')
    fig.colorbar(strm.lines)
    ax1.set_title('Variation de la couleur')

    #  changer la largeur le long d'une ligne
    ax2 = fig.add_subplot(gs[1, 0])
    lw = 5*vitesse / vitesse.max()
    ax2.streamplot(X, Y, U, V, density=0.6, color='orange', linewidth=lw)
    ax2.set_title('Variation de la largeur')

    plt.tight_layout()
    plt.show()


# d'après le premier projet numérique
def affiche_courbes_niveaux(x0 = alpha/beta, y0 = gamma/delta):
    for i in range(20):
        tableau = courbe_niveau(H, x0 + i*eps, y0 + i*eps)
        plt.plot(tableau[0], tableau[1], color='red')
    plt.show()


def solve_euler_explicit(f, x0, y0, dt, t0, tf):
    # on a x_(n+1) = x_n + dt*f(x_n)

    n = int((tf - t0)/dt)

    res = np.empty(n, dtype = float)

    res[0] = x0

    for i in range(1, n):
        res[i] = res[i-1] + dt*f(res[i - 1])

    return res


affiche_courbes_niveaux()
