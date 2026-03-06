import json
import os
import numpy as np

from src.grids import get_grilles
from src.astar import astar, ucs, greedy, weighted_astar, manhattan, null_heuristic
from src.markov import (
    construire_politique, construire_matrice_transition,
    evolution_distribution, proba_goal_au_temps,
    analyse_absorption, simulation_monte_carlo,
)

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


def _save_json(data, filename):
    path = os.path.join(RESULTS_DIR, filename)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, default=str)
    print(f"  -> Sauvegardé : {path}")


# E.1 : Comparer UCS vs Greedy vs A* sur 3 grilles
def experience_E1():
    print("=== E.1 : UCS vs Greedy vs A* ===")
    grilles = get_grilles()
    resultats = {}

    for nom, (grid, start, goal) in grilles.items():
        resultats[nom] = {}
        for algo_name, algo_func in [("UCS", ucs), ("Greedy", greedy), ("A*", astar)]:
            if algo_name == "UCS":
                path, cost, expanded, dt, max_open = algo_func(grid, start, goal)
            else:
                path, cost, expanded, dt, max_open = algo_func(grid, start, goal)

            resultats[nom][algo_name] = {
                "cout": cost,
                "noeuds_developpes": expanded,
                "temps_s": round(dt, 6),
                "taille_max_open": max_open,
                "longueur_chemin": len(path) if path else 0,
                "chemin": [list(p) for p in path] if path else None,
            }
            print(f"  {nom}/{algo_name}: coût={cost}, expand={expanded}, t={dt:.4f}s")

    _save_json(resultats, "E1_comparaison_algos.json")
    return resultats


# E.2 : A* fixé, varier epsilon, mesurer proba GOAL (Markov)
def experience_E2(epsilons=None, n_sim=2000, n_steps_distrib=200):
    if epsilons is None:
        epsilons = [0.0, 0.1, 0.2, 0.3]
    print("=== E.2 : Impact de epsilon ===")
    grilles = get_grilles()
    resultats = {}

    for nom, (grid, start, goal) in grilles.items():
        path, cost, expanded, dt, max_open = astar(grid, start, goal)
        if path is None:
            continue
        politique = construire_politique(grid, path)
        resultats[nom] = {"cout_astar": cost, "longueur_chemin": len(path), "epsilons": {}}

        for eps in epsilons:
            P, etats_libres, etat_to_idx, GOAL_IDX, FAIL_IDX = \
                construire_matrice_transition(grid, politique, eps, goal)

            # vérification stochastique
            assert np.allclose(P.sum(axis=1), 1.0), "P non stochastique !"

            # distribution initiale
            pi0 = np.zeros(P.shape[0])
            pi0[etat_to_idx[start]] = 1.0

            # évolution pi^(n)
            distribs = evolution_distribution(P, pi0, n_steps_distrib)
            proba_goal = proba_goal_au_temps(distribs, GOAL_IDX)

            # absorption
            N, B, t_abs = analyse_absorption(P, etats_libres, etat_to_idx, GOAL_IDX, FAIL_IDX)
            idx_start = etat_to_idx[start]
            proba_abs_goal = float(B[idx_start, 0])
            proba_abs_fail = float(B[idx_start, 1])
            temps_moyen_abs = float(t_abs[idx_start])

            # simulation Monte Carlo
            mc = simulation_monte_carlo(grid, start, goal, politique, eps, n_sim=n_sim)

            resultats[nom]["epsilons"][str(eps)] = {
                "proba_goal_curve": [round(float(p), 6) for p in proba_goal],
                "proba_abs_goal": round(proba_abs_goal, 6),
                "proba_abs_fail": round(proba_abs_fail, 6),
                "temps_moyen_absorption": round(temps_moyen_abs, 2),
                "mc_proba_succes": mc["proba_succes"],
                "mc_temps_moyen": round(mc["temps_moyen"], 2) if mc["temps_moyen"] != float('inf') else "inf",
                "mc_temps_std": round(mc["temps_std"], 2),
            }
            print(f"  {nom}/eps={eps}: P(GOAL)_abs={proba_abs_goal:.4f}, "
                  f"MC={mc['proba_succes']:.4f}, t_moy={temps_moyen_abs:.1f}")

    _save_json(resultats, "E2_impact_epsilon.json")
    return resultats


# E.3 : h=0 vs Manhattan – comparaison des expansions
def experience_E3():
    print("=== E.3 : h=0 vs Manhattan ===")
    grilles = get_grilles()
    resultats = {}

    for nom, (grid, start, goal) in grilles.items():
        resultats[nom] = {}
        for h_name, h_func in [("h=0 (UCS)", null_heuristic), ("Manhattan", manhattan)]:
            path, cost, expanded, dt, max_open = astar(grid, start, goal, h_func=h_func)
            resultats[nom][h_name] = {
                "cout": cost,
                "noeuds_developpes": expanded,
                "temps_s": round(dt, 6),
                "taille_max_open": max_open,
                "longueur_chemin": len(path) if path else 0,
            }
            print(f"  {nom}/{h_name}: expand={expanded}, coût={cost}")

    _save_json(resultats, "E3_heuristiques.json")
    return resultats


# E.4 : Weighted A* – compromis vitesse vs optimalité
def experience_E4(weights=None):
    if weights is None:
        weights = [1.0, 1.5, 2.0, 3.0, 5.0]
    print("=== E.4 : Weighted A* ===")
    grilles = get_grilles()
    resultats = {}

    for nom, (grid, start, goal) in grilles.items():
        resultats[nom] = {}
        for w in weights:
            path, cost, expanded, dt, max_open = weighted_astar(grid, start, goal, w=w)
            resultats[nom][f"w={w}"] = {
                "poids": w,
                "cout": cost,
                "noeuds_developpes": expanded,
                "temps_s": round(dt, 6),
                "taille_max_open": max_open,
                "longueur_chemin": len(path) if path else 0,
                "chemin": [list(p) for p in path] if path else None,
            }
            print(f"  {nom}/w={w}: coût={cost}, expand={expanded}")

    _save_json(resultats, "E4_weighted_astar.json")
    return resultats


# Lancer toutes les expériences
def run_all():
    print("Lancement de toutes les expériences...\n")
    experience_E1()
    print()
    experience_E2()
    print()
    experience_E3()
    print()
    experience_E4()
    print("\nTerminé ! Résultats dans le dossier results/")


if __name__ == "__main__":
    run_all()
