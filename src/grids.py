import numpy as np


# --- Définition des grilles : 0 = libre, 1 = obstacle ---

def grille_facile():
    """Grille 8x8, deux murs avec ouvertures opposées."""
    grid = np.zeros((8, 8), dtype=int)
    # mur col 3, ouverture en bas (row 6)
    for r in range(0, 6):
        grid[r, 3] = 1
    # mur col 5, ouverture en haut (row 1)
    for r in range(1, 8):
        grid[r, 5] = 1
    grid[1, 5] = 0
    # obstacles supplémentaires
    grid[6, 1] = 1
    grid[7, 4] = 1
    grid[1, 7] = 1
    start = (0, 0)
    goal = (7, 7)
    return grid, start, goal


def grille_moyenne():
    """Grille 12x12, murs asymétriques forçant des détours."""
    grid = np.zeros((12, 12), dtype=int)
    # mur col 3, ouverture à row 9
    for r in range(0, 11):
        grid[r, 3] = 1
    grid[9, 3] = 0
    # mur col 8, ouverture à row 2
    for r in range(1, 12):
        grid[r, 8] = 1
    grid[2, 8] = 0
    # obstacles dispersés
    for r, c in [(1,6),(5,1),(5,2),(10,5),(10,6),(7,10)]:
        grid[r, c] = 1
    start = (0, 0)
    goal = (11, 11)
    return grid, start, goal


def grille_difficile():
    """Grille 15x15, deux grands murs avec passages éloignés."""
    grid = np.zeros((15, 15), dtype=int)
    # mur col 5, ouverture à row 12
    for r in range(0, 12):
        grid[r, 5] = 1
    # mur col 9, ouverture à row 2
    for r in range(2, 15):
        grid[r, 9] = 1
    grid[2, 9] = 0
    # obstacles supplémentaires
    for r, c in [(1,2),(1,3),(13,3),(13,4),(3,7),(3,8),(7,11),(7,12),(10,1),(11,13),(12,7)]:
        grid[r, c] = 1
    start = (0, 0)
    goal = (14, 14)
    return grid, start, goal


def get_grilles():
    """Retourne un dict {nom: (grid, start, goal)}."""
    return {
        "facile": grille_facile(),
        "moyenne": grille_moyenne(),
        "difficile": grille_difficile(),
    }


def voisins(grid, pos):
    """Retourne les voisins accessibles (4-connexité)."""
    rows, cols = grid.shape
    r, c = pos
    for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
        nr, nc = r + dr, c + dc
        if 0 <= nr < rows and 0 <= nc < cols and grid[nr, nc] == 0:
            yield (nr, nc)


# directions cardinales utilisées par le modèle Markov
ACTIONS = [(-1,0),(1,0),(0,-1),(0,1)]  # haut, bas, gauche, droite

def action_vers(pos, next_pos):
    """Retourne l'index d'action pour aller de pos à next_pos."""
    dr = next_pos[0] - pos[0]
    dc = next_pos[1] - pos[1]
    return ACTIONS.index((dr, dc))


# déviations latérales par action
LATERAUX = {
    0: [2, 3],  # haut -> gauche, droite
    1: [2, 3],  # bas  -> gauche, droite
    2: [0, 1],  # gauche -> haut, bas
    3: [0, 1],  # droite -> haut, bas
}
