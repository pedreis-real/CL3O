'''
================================================================================
CL3O - Composite Lifting Surface Structural Sizing & Optimization.
Differential Evolution Data Containers Module.

Dataclass containers for the Differential Evolution optimizer: the run
configuration snapshot (OptData), the per-generation history (HistoryData) and
the decoded per-control-point design variables (OptVars). Split out of de_opt.py
so the data schema imports without pulling in the solver; de_opt re-exports these
names for backward compatibility (including pickle resolution of archived runs).

@ CL3O Authors - MIT License
================================================================================
'''

# ================ PyLib imports ================
from dataclasses import dataclass, field

import numpy as np

# ================ Module imports ================

# Constants
from cl3o.Constants import DE_HYPERPAR


@dataclass
class OptData:
    '''
    DE setup snapshot: bounds, hyper-parameters, initial population.

    Property    Size        Description
    --------    --------    ----------------------------------------
    lo          (D,)        Lower bound per design variable     
    hi          (D,)        Upper bound per design variable     
    NP          (1,)        Population size                     
    CR          (1,)        Crossover probability [0, 1]        
    F           (1,)        Differential weight                 
    lmbda       (1,)        Best-attraction weight              
    k_max       (1,)        Maximum number of generations       
    seed        (1,)        RNG seed for reproducibility        
    D           (1,)        Number of design variables          
    X0          (NP, D)     Initial population (seeded LHS-like)
    '''
    lo             : np.ndarray = field(default_factory=lambda: np.zeros(0))
    hi             : np.ndarray = field(default_factory=lambda: np.zeros(0))
    NP             : int        = DE_HYPERPAR['NP']
    CR             : float      = DE_HYPERPAR['CR']
    F              : float      = DE_HYPERPAR['F']
    lmbda          : float      = DE_HYPERPAR['lambda']
    k_max          : int        = DE_HYPERPAR['k_max']
    seed           : int        = DE_HYPERPAR['seed']
    tol            : float      = DE_HYPERPAR['std_tol']
    stall_patience : int        = DE_HYPERPAR['stall_patience']
    D              : int        = 0
    X0             : np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))


@dataclass
class HistoryData:
    '''
    Per-generation history of the DE run.

    Property        Size            Description
    ------------    ------------    -----------------------------------
    ng              (1,)            Number of generations executed
    D               (1,)            Design-space dimension        
    best_X          (ng + 1, D)     Best design at each generation     
    best_f          (ng + 1,)       Fitness of best design at each gen.
    mean_f          (ng + 1,)       Population mean fitness            
    std_f           (ng + 1,)       Population fitness std             
    feasible_X      (D,)            Best feasible design vector        
    feasible_f      (1,)            Fitness of best feasible design    
    '''
    ng         : int        = 0
    D          : int        = 0
    best_X     : np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
    best_f     : np.ndarray = field(default_factory=lambda: np.zeros(0))
    mean_f     : np.ndarray = field(default_factory=lambda: np.zeros(0))
    std_f      : np.ndarray = field(default_factory=lambda: np.zeros(0))
    feasible_X : np.ndarray = field(default_factory=lambda: np.zeros(0))
    feasible_f : float      = float('inf')


@dataclass
class OptVars:
    '''
    Container exposing per-cpt design-variables.

    The DE loop produces a flat design vector X; this container
    de-serialises it into the six continuous and eight discrete
    per-cpt arrays that SectionBuilder expects.

    Property    Size        Description                                 Units
    --------    --------    ----------------------------------------    --------
    xw1         (ncpt,)     Rear wing spar position                     - %c
    xw2         (ncpt,)     Aft  wing spar position                     - %c
    bf1_root    (1,)        Upper rear flange width at root             - %c
    bf2_root    (1,)        Lower rear flange width at root             - %c
    bf3_root    (1,)        Upper aft  flange width at root             - %c
    bf4_root    (1,)        Lower aft  flange width at root             - %c
    tpr         (ncpt-1,)   Flange taper relative to root station       - 0-1
    bf1         (ncpt-1,)   Upper rear flange width per non-root CP     - %c
    bf2         (ncpt-1,)   Lower rear flange width per non-root CP     - %c
    bf3         (ncpt-1,)   Upper aft  flange width per non-root CP     - %c
    bf4         (ncpt-1,)   Lower aft  flange width per non-root CP     - %c
    ls1         (ncpt,)     Skin layup, from LE up to xw1               - index
    ls2         (ncpt,)     Skin layup, from xw1 up to TE               - index
    lw1         (ncpt,)     Rear wing web layup                         - index
    lw2         (ncpt,)     Aft  wing web layup                         - index
    lf1         (ncpt,)     Upper rear flange layup                     - index
    lf2         (ncpt,)     Lower rear flange layup                     - index
    lf3         (ncpt,)     Upper aft  flange layup                     - index
    lf4         (ncpt,)     Lower aft  flange layup                     - index
    '''
    xw1 : np.ndarray = field(default_factory=lambda: np.zeros(0))
    xw2 : np.ndarray = field(default_factory=lambda: np.zeros(0))

    bf1_root : float      = 0.0
    bf2_root : float      = 0.0
    bf3_root : float      = 0.0
    bf4_root : float      = 0.0
    tpr      : np.ndarray = field(default_factory=lambda: np.zeros(0))

    bf1 : np.ndarray = field(default_factory=lambda: np.zeros(0))
    bf2 : np.ndarray = field(default_factory=lambda: np.zeros(0))
    bf3 : np.ndarray = field(default_factory=lambda: np.zeros(0))
    bf4 : np.ndarray = field(default_factory=lambda: np.zeros(0))

    ls1 : np.ndarray = field(default_factory=lambda: np.zeros(0))
    ls2 : np.ndarray = field(default_factory=lambda: np.zeros(0))
    lw1 : np.ndarray = field(default_factory=lambda: np.zeros(0))
    lw2 : np.ndarray = field(default_factory=lambda: np.zeros(0))
    lf1 : np.ndarray = field(default_factory=lambda: np.zeros(0))
    lf2 : np.ndarray = field(default_factory=lambda: np.zeros(0))
    lf3 : np.ndarray = field(default_factory=lambda: np.zeros(0))
    lf4 : np.ndarray = field(default_factory=lambda: np.zeros(0))
