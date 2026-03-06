import heapq
import time
from src.grids import voisins


def manhattan(pos, goal):
    return abs(pos[0] - goal[0]) + abs(pos[1] - goal[1])


def null_heuristic(pos, goal):
    return 0


def recherche_generique(grid, start, goal, h_func=manhattan, w=1.0, mode="astar"):
    """
    Recherche sur grille avec OPEN (heap) et CLOSED (set).
    mode: 'astar' -> f=g+w*h, 'ucs' -> f=g, 'greedy' -> f=h
    Retourne (chemin, coût, noeuds_developpes, temps, taille_max_open).
    """
    t0 = time.perf_counter()

    if mode == "ucs":
        f = lambda g, pos: g
    elif mode == "greedy":
        f = lambda g, pos: h_func(pos, goal)
    else:
        f = lambda g, pos: g + w * h_func(pos, goal)

    # (f_val, compteur, g_val, position, parent)
    counter = 0
    open_list = []
    heapq.heappush(open_list, (f(0, start), counter, 0, start, None))
    closed = set()
    came_from = {}
    g_score = {start: 0}
    max_open = 1
    nodes_expanded = 0

    while open_list:
        max_open = max(max_open, len(open_list))
        f_val, _, g_val, current, parent = heapq.heappop(open_list)

        if current in closed:
            continue

        came_from[current] = parent
        closed.add(current)
        nodes_expanded += 1

        if current == goal:
            path = []
            node = current
            while node is not None:
                path.append(node)
                node = came_from[node]
            path.reverse()
            dt = time.perf_counter() - t0
            return path, g_val, nodes_expanded, dt, max_open

        for nb in voisins(grid, current):
            if nb in closed:
                continue
            new_g = g_val + 1
            if new_g < g_score.get(nb, float('inf')):
                g_score[nb] = new_g
                counter += 1
                heapq.heappush(open_list, (f(new_g, nb), counter, new_g, nb, current))

    dt = time.perf_counter() - t0
    return None, float('inf'), nodes_expanded, dt, max_open


def astar(grid, start, goal, h_func=manhattan):
    return recherche_generique(grid, start, goal, h_func=h_func, mode="astar")


def ucs(grid, start, goal):
    return recherche_generique(grid, start, goal, mode="ucs")


def greedy(grid, start, goal, h_func=manhattan):
    return recherche_generique(grid, start, goal, h_func=h_func, mode="greedy")


def weighted_astar(grid, start, goal, w=1.5, h_func=manhattan):
    return recherche_generique(grid, start, goal, h_func=h_func, w=w, mode="astar")
