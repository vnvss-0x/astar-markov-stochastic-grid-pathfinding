import numpy as np
from src.grids import voisins, ACTIONS, LATERAUX, action_vers


def construire_politique(grid, chemin):
    """
    À partir d'un chemin A*, crée une politique : état -> index d'action.
    Pour les états hors du chemin, on choisit l'action qui rapproche du goal.
    """
    politique = {}
    for i in range(len(chemin) - 1):
        politique[chemin[i]] = action_vers(chemin[i], chemin[i + 1])
    return politique


def _appliquer_action(grid, pos, action_idx):
    """Retourne la position résultante d'une action. Si obstacle/mur, reste sur place."""
    dr, dc = ACTIONS[action_idx]
    nr, nc = pos[0] + dr, pos[1] + dc
    rows, cols = grid.shape
    if 0 <= nr < rows and 0 <= nc < cols and grid[nr, nc] == 0:
        return (nr, nc)
    return pos  # bloqué


def construire_matrice_transition(grid, politique, epsilon, goal):
    """
    Construit la matrice P sur les états libres + GOAL + FAIL.
    GOAL et FAIL sont des états absorbants.
    États hors politique -> FAIL (absorbant).
    """
    rows, cols = grid.shape
    etats_libres = [(r, c) for r in range(rows) for c in range(cols) if grid[r, c] == 0]
    
    # index : états libres + GOAL_abs + FAIL_abs
    etat_to_idx = {s: i for i, s in enumerate(etats_libres)}
    n = len(etats_libres)
    GOAL_IDX = n
    FAIL_IDX = n + 1
    total = n + 2

    P = np.zeros((total, total))

    # GOAL et FAIL absorbants
    P[GOAL_IDX, GOAL_IDX] = 1.0
    P[FAIL_IDX, FAIL_IDX] = 1.0

    for s in etats_libres:
        i = etat_to_idx[s]

        if s == goal:
            P[i, GOAL_IDX] = 1.0
            continue

        if s not in politique:
            # pas de politique -> va vers FAIL
            P[i, FAIL_IDX] = 1.0
            continue

        action = politique[s]
        lateraux = LATERAUX[action]

        # action principale avec proba (1 - epsilon)
        dest_main = _appliquer_action(grid, s, action)
        j_main = etat_to_idx[dest_main]
        P[i, j_main] += (1 - epsilon)

        # déviations latérales avec proba epsilon/2 chacune
        for lat_action in lateraux:
            dest_lat = _appliquer_action(grid, s, lat_action)
            j_lat = etat_to_idx[dest_lat]
            P[i, j_lat] += epsilon / 2

    return P, etats_libres, etat_to_idx, GOAL_IDX, FAIL_IDX


def evolution_distribution(P, pi0, n_steps):
    """Calcule pi^(n) = pi^(0) * P^n pour n = 0..n_steps."""
    distributions = [pi0.copy()]
    pi = pi0.copy()
    for _ in range(n_steps):
        pi = pi @ P
        distributions.append(pi.copy())
    return distributions


def proba_goal_au_temps(distributions, goal_idx):
    """Retourne la proba d'être dans GOAL à chaque pas."""
    return [pi[goal_idx] for pi in distributions]


def analyse_absorption(P, etats_libres, etat_to_idx, GOAL_IDX, FAIL_IDX):
    """
    Décomposition en états transitoires / absorbants.
    Retourne la matrice fondamentale N = (I-Q)^{-1},
    les probas d'absorption B = N*R, et le temps moyen t = N*1.
    """
    n = len(etats_libres)
    total = n + 2
    # indices transitoires = 0..n-1, absorbants = GOAL_IDX, FAIL_IDX
    trans = list(range(n))
    absorb = [GOAL_IDX, FAIL_IDX]

    Q = P[np.ix_(trans, trans)]
    R = P[np.ix_(trans, absorb)]

    I = np.eye(len(trans))
    N = np.linalg.inv(I - Q)  # matrice fondamentale
    B = N @ R  # probas d'absorption (col 0 = GOAL, col 1 = FAIL)
    t = N @ np.ones(len(trans))  # temps moyen avant absorption

    return N, B, t


def simulation_trajectoire(grid, start, goal, politique, epsilon, max_steps=500):
    """Simule une trajectoire Markov. Retourne (succes, nb_pas, trajectoire)."""
    pos = start
    traj = [pos]

    for step in range(max_steps):
        if pos == goal:
            return True, step, traj

        if pos not in politique:
            return False, step, traj

        action = politique[pos]
        lateraux = LATERAUX[action]

        r = np.random.random()
        if r < (1 - epsilon):
            chosen = action
        elif r < (1 - epsilon / 2):
            chosen = lateraux[0]
        else:
            chosen = lateraux[1]

        pos = _appliquer_action(grid, pos, chosen)
        traj.append(pos)

    return False, max_steps, traj


def simulation_monte_carlo(grid, start, goal, politique, epsilon, n_sim=1000, max_steps=500):
    """Lance n_sim simulations et retourne les statistiques."""
    succes = 0
    temps_atteinte = []

    for _ in range(n_sim):
        ok, steps, _ = simulation_trajectoire(grid, start, goal, politique, epsilon, max_steps)
        if ok:
            succes += 1
            temps_atteinte.append(steps)

    proba_succes = succes / n_sim
    temps_moyen = np.mean(temps_atteinte) if temps_atteinte else float('inf')
    temps_std = np.std(temps_atteinte) if temps_atteinte else 0.0

    return {
        "proba_succes": proba_succes,
        "temps_moyen": temps_moyen,
        "temps_std": temps_std,
        "n_succes": succes,
        "n_sim": n_sim,
        "temps_atteinte": temps_atteinte,
    }
