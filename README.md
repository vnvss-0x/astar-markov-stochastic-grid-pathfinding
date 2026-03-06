# A* et Chaine de Markov -- Planification stochastique sur grille

Ce projet combine des **algorithmes de recherche heuristique** (A\*, UCS, Greedy, Weighted A\*) avec une **analyse par chaine de Markov** pour etudier la planification sous incertitude sur des grilles 2D. Un chemin deterministe est d'abord calcule, puis evalue sous un modele de transition stochastique ou l'agent peut devier lateralement avec une probabilite epsilon a chaque pas.

---

## Structure du projet

```
.
├── src/
│   ├── grids.py          # Definition des grilles (3 niveaux) et fonctions
│   ├── astar.py          # Recherche generique : A*, UCS, Greedy, Weighted A*
│   ├── markov.py         # Chaine de Markov : matrice de transition, absorption, Monte Carlo
│   └── experiments.py    # Lancement des experiences E1-E4, export JSON
├── notebooks/
│   └── analyse_resultats.ipynb   # Notebook Jupyter avec figures et analyse
├── results/              # Donnees JSON et figures PNG generees
├── report/               # Rapport
├── requirements.txt
└── README.md
```

## Algorithmes

### Recherche (src/astar.py)

Une fonction generique unique `recherche_generique()` implemente les quatre variantes via un parametre `mode` :

| Algorithme    | Priorite f(n)       | Optimal | Remarques                              |
|---------------|---------------------|---------|----------------------------------------|
| UCS           | g(n)                | Oui     | Explore tous les noeuds accessibles    |
| Greedy        | h(n)                | Non     | Rapide mais peut trouver des chemins plus longs |
| A*            | g(n) + h(n)         | Oui     | Optimal avec h admissible              |
| Weighted A*   | g(n) + w * h(n)     | Non     | Compromis optimalite / vitesse         |

Heuristique utilisee : distance de Manhattan (admissible et consistante sur grilles 4-connexes).

### Chaine de Markov (src/markov.py)

A partir d'un chemin deterministe (politique), le modele stochastique introduit des deviations laterales :

- Avec probabilite `1 - epsilon` : l'agent suit l'action prevue.
- Avec probabilite `epsilon / 2` chacune : l'agent devie vers l'une des deux directions laterales.

Calculs principaux :
- **Matrice de transition P** avec deux etats absorbants (GOAL et FAIL).
- **Matrice fondamentale** N = (I - Q)^{-1} pour l'analyse d'absorption.
- **Probabilites d'absorption** B = N * R (probabilite d'atteindre GOAL vs FAIL).
- **Temps moyen d'absorption** t = N * 1.
- **Simulation Monte Carlo** pour validation empirique.

## Grilles

Trois grilles de difficulte croissante, chacune comportant deux murs verticaux avec des ouvertures asymetriques qui creent des chemins optimaux et sous-optimaux distincts :

| Grille     | Taille | Cellules libres | Obstacles | Cout optimal (A*) |
|------------|--------|-----------------|-----------|--------------------|
| Facile     | 8x8    | 49              | 15        | 24                 |
| Moyenne    | 12x12  | 118             | 26        | 36                 |
| Difficile  | 15x15  | 190             | 35        | 48                 |

La conception des grilles garantit que Greedy trouve un chemin sous-optimal sur chaque grille, rendant les comparaisons algorithmiques significatives.

## Experiences

### E.1 -- Comparaison des algorithmes (UCS vs Greedy vs A*)

Compare le cout, les noeuds developpes, le temps d'execution et la taille maximale de OPEN sur les trois grilles. Inclut une visualisation 3x3 des chemins trouves.

### E.2 -- Impact de l'incertitude (epsilon)

Fixe le chemin optimal A* et varie epsilon dans {0, 0.1, 0.2, 0.3}. Mesure :
- P(GOAL) analytique via l'absorption matricielle.
- P(GOAL) empirique via Monte Carlo (N=2000).
- Evolution temporelle de la distribution.
- Temps moyen d'absorption.

### E.3 -- Comparaison des heuristiques (h=0 vs Manhattan)

Compare A* avec une heuristique nulle (equivalent a UCS) contre la distance de Manhattan. Mesure la reduction du nombre de noeuds developpes.

### E.4 -- Compromis Weighted A*

Varie le poids w dans {1.0, 1.5, 2.0, 3.0, 5.0}. Montre le compromis entre la reduction des noeuds developpes et l'augmentation du cout du chemin.

## Installation et utilisation

### Pre-requis

- Python 3.10+
- numpy
- matplotlib

### Installation

```bash
pip install -r requirements.txt
```

### Lancer toutes les experiences

```bash
python -m src.experiments
```

Cela genere quatre fichiers JSON et affiche les resultats dans la console. Les sorties sont sauvegardees dans le dossier `results/`.

### Analyse dans le notebook

Ouvrir `notebooks/analyse_resultats.ipynb` dans Jupyter ou VS Code. Le notebook charge les resultats JSON et produit toutes les figures (sauvegardees en PNG dans `results/`).

## Sorties generees

### Donnees JSON

| Fichier                       | Contenu                                              |
|-------------------------------|------------------------------------------------------|
| E1_comparaison_algos.json     | Cout, noeuds developpes, temps pour UCS/Greedy/A*    |
| E2_impact_epsilon.json        | Probabilites d'absorption, stats MC par epsilon      |
| E3_heuristiques.json          | Comparaison h=0 vs Manhattan                         |
| E4_weighted_astar.json        | Resultats Weighted A* pour chaque poids w            |

### Figures

| Figure                          | Description                                        |
|---------------------------------|----------------------------------------------------|
| fig_grilles.png                 | Les 3 grilles avec les chemins optimaux A*         |
| fig_E1_comparaison.png          | Diagrammes en barres : cout, expansions, temps, OPEN |
| fig_E1_chemins.png              | Sous-graphes 3x3 : chemins par algorithme et grille |
| fig_E2_proba_goal.png           | P(GOAL) analytique vs Monte Carlo                  |
| fig_E2_evolution_temporelle.png | Evolution de la distribution au cours du temps     |
| fig_E2_temps_absorption.png     | Temps moyen d'absorption vs epsilon                |
| fig_E3_heuristiques.png         | Impact de l'heuristique sur les noeuds developpes  |
| fig_E4_weighted_astar.png       | Noeuds developpes et cout vs poids w               |
| fig_monte_carlo_analyse.png     | Trajectoires MC et distribution des pas            |

## Resultats principaux

- **A\*** trouve les chemins optimaux sur toutes les grilles avec significativement moins d'expansions que UCS.
- **Greedy** est le plus rapide mais trouve systematiquement des chemins sous-optimaux (8-11% plus longs).
- **P(GOAL) chute fortement avec epsilon** : de 1.0 (deterministe) a quasi 0 pour epsilon=0.3, surtout sur les grilles difficiles.
- **Les resultats analytiques et Monte Carlo concordent etroitement**, validant le modele de Markov.
- **Weighted A\*** offre un compromis reglable : un w plus eleve reduit les expansions au prix de la qualite du chemin.

## Auteur

Anass MAKHLOUK
