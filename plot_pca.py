import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import decomposition
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.stats import pearsonr
sns.set()


def plot_features(pca, Z, x_y, features, figsize=(10,10), illustrative_features=[], illustrative_names=[]) : 
    """
    Affiche le graphe des correlations

    Positional arguments : 
    -----------------------------------
    pca : sklearn.decomposition.PCA : notre objet PCA qui a été fit
    Z : ndarray (n_samples, n_features) : Les données ayant servi à fitter le PCA
    x_y : list ou tuple : le couple x,y des plans à afficher, exemple [0,1] pour F1, F2
    features : list ou tuple : la liste des features (ie des dimensions) à représenter
    illustrative_features : ndarray (n_samples, m>1) : les features qui n'ont pas servi à établir le PCA mais
                                                       que l'on souhaite représenter sur le cercle de corrélations
    illustrative_names : list ou tuple (len = m) : noms des features illustratives
    figsize : list ou tuple : couple width, height qui définit la taille de la figure en inches, default = [10,8]
    """
    
    # Extrait x et y 
    x,y=x_y

    # On définit la forme de la figure si elle n'a pas été donnée
    if not figsize: 
        figsize = (10,10)

    # Taille de l'image (en inches)
    fig, ax = plt.subplots(figsize=figsize)

    # Pour chaque composante active: 
    for i in range(0, pca.components_.shape[1]):

        # Les flèches
        ax.arrow(0, 0, pca.components_[x, i], pca.components_[y, i], head_width=0.04, head_length=0.04, width=0.02)

        # Les labels
        plt.text(pca.components_[x, i] + 0.005, pca.components_[y, i] + 0.005, features[i], fontsize='9')
    
    if len(illustrative_features)>0:
        # Coordonnées des individus dans les axes principaux de l'acp
        Z_new = pca.transform(Z)        # ndarray(n_samples, n_components)
        
        # Coordonées des individus sur les axes Fx et Fy
        zx = Z_new[:,x]         # ndarray(n_samples,)
        zy = Z_new[:,y]         # ndarray(n_samples,)
        
        # Pour chaque composante illustrative:  
        for j in range(illustrative_features.shape[1]):
            
            # Coordonnées de la variable illustrative sur le plan factoriel
            x_var, _ = pearsonr(illustrative_features[:,j], zx)
            y_var, _ = pearsonr(illustrative_features[:,j], zy)
            
            # Les flèches
            ax.arrow(0, 0, x_var, y_var, head_width=0.03, head_length=0.03, width=0.02, linestyle="--", color="orange")

            # Les labels
            plt.text(x_var+0.01, y_var+0.01, illustrative_names[j], fontsize='9')
        
    # Affichage des lignes horizontales et verticales
    plt.plot([-1, 1], [0, 0], color='grey', ls='--')
    plt.plot([0, 0], [-1, 1], color='grey', ls='--')

    # Nom des axes, avec le pourcentage d'inertie expliqué
    plt.xlabel('F{} ({}%)'.format(x+1, round(100*pca.explained_variance_ratio_[x],1)))
    plt.ylabel('F{} ({}%)'.format(y+1, round(100*pca.explained_variance_ratio_[y],1)))

    # Titre du graphe
    plt.title("Cercle des corrélations (F{} et F{})".format(x+1, y+1))

    # Le cercle 
    an = np.linspace(0, 2 * np.pi, 100)
    plt.plot(np.cos(an), np.sin(an))  # Add a unit circle for scale

    # Axes et display
    plt.axis('equal')
    plt.show(block=False)



def plot_individus(X_projected, x_y, pca=None, labels = None, clusters=None, alpha=1, 
                             figsize=(10,10), marker="." ):
    """
    Affiche la projection des individus

    Positional arguments : 
    -------------------------------------
    X_projected : np.array, pd.DataFrame, list of list : la matrice des points projetés
    x_y : list ou tuple : le couple x,y des plans à afficher, exemple [0,1] pour F1, F2

    Optional arguments : 
    -------------------------------------
    pca : sklearn.decomposition.PCA : un objet PCA qui a été fit, cela nous permettra d'afficher la variance de chaque composante, default = None
    labels : list ou tuple : les labels des individus à projeter, default = None
    clusters : list ou tuple : la liste des clusters auquel appartient chaque individu, default = None
    alpha : float in [0,1] : paramètre de transparence, 0=100% transparent, 1=0% transparent, default = 1
    figsize : list ou tuple : couple width, height qui définit la taille de la figure en inches, default = [10,8] 
    marker : str : le type de marker utilisé pour représenter les individus, points croix etc etc, default = "."
    """

    # Transforme X_projected en np.array
    X_ = np.array(X_projected)

    # On définit la forme de la figure si elle n'a pas été donnée
    if not figsize: 
        figsize = (10,10)

    # On gère les labels
    if  labels is None : 
        labels = []
    try : 
        len(labels)
    except Exception as e : 
        raise e

    # On vérifie la variable axis 
    if not len(x_y) ==2 : 
        raise AttributeError("2 axes sont demandées")   
    if max(x_y )>= X_.shape[1] : 
        raise AttributeError("la variable axis n'est pas bonne")   

    # on définit x et y 
    x, y = x_y

    # Initialisation de la figure       
    fig, ax = plt.subplots(1, 1, figsize=figsize)

    # On vérifie s'il y a des clusters ou non
    c = None if clusters is None else clusters
 
    # Les points    
    sns.scatterplot(data=None, x=X_[:, x], y=X_[:, y], hue=c)

    # Si la variable pca a été fournie, on peut calculer le % de variance de chaque axe 
    if pca : 
        v1 = str(round(100*pca.explained_variance_ratio_[x]))  + " %"
        v2 = str(round(100*pca.explained_variance_ratio_[y]))  + " %"
    else : 
        v1=v2= ''

    # Nom des axes, avec le pourcentage d'inertie expliqué
    ax.set_xlabel(f'F{x+1} {v1}')
    ax.set_ylabel(f'F{y+1} {v2}')

    # Valeur x max et y max
    x_max = np.abs(X_[:, x]).max() *1.1
    y_max = np.abs(X_[:, y]).max() *1.1

    # On borne x et y 
    ax.set_xlim(left=-x_max, right=x_max)
    ax.set_ylim(bottom= -y_max, top=y_max)

    # Affichage des lignes horizontales et verticales
    plt.plot([-x_max, x_max], [0, 0], color='grey', alpha=0.8)
    plt.plot([0,0], [-y_max, y_max], color='grey', alpha=0.8)

    # Affichage des labels des points
    if len(labels) : 
        # Pour chaque point
        for i,(_x,_y) in enumerate(X_[:,[x,y]]):
            plt.text(_x, _y+0.01, labels[i], fontsize='9', ha='center',va='center') 

    # Titre et display
    plt.title(f"Projection des individus (sur F{x+1} et F{y+1})")
    plt.show()