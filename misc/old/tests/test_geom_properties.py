'''
Smoke test for GeomPropCalculator.

Loads the Wortmann FX 63-137 airfoil and runs the full pipeline with
representative DA-62 wing station parameters to verify no crashes.
'''

import sys
from pathlib import Path
import json
import numpy as np

# Bootstrap paths
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))

from geometry.geom_properties import GeomPropCalculator, GeomData


def load_airfoil():
    '''Load the Wortmann FX 63-137 airfoil data from JSON.'''
    path = _ROOT / "data" / "airfoils" / "wortmannfx63137_AirfoilData.json"
    with open(path) as f:
        d = json.load(f)
    return {k: np.array(v) for k, v in d.items()}


def test_smoke():
    '''Run full pipeline and check output shapes and signs.'''
    afl = load_airfoil()

    # Representative station: chord=1500mm, no twist, Y=2000mm
    # Spar positions at 25% and 65% chord
    # Uniform material: t=1mm, G=5000MPa, E1=60000MPa, E2=5000MPa
    calc = GeomPropCalculator(
        x_upper  = afl['x_upper'],
        y_upper  = afl['y_upper'],
        x_lower  = afl['x_lower'],
        y_lower  = afl['y_lower'],
        x_camber = afl['x_camber'],
        y_camber = afl['y_camber'],
        chord     = 1500.0,
        twist = 0.0,
        Y_sta     = 2000.0,
        xw1       = 0.25,
        xw2       = 0.65,
        bf1       = 20.0,
        bf2       = 20.0,
        bf3       = 15.0,
        bf4       = 15.0,
        t_seg     = np.full(7, 1.0),
        G_seg     = np.full(7, 5000.0),
        E1_seg    = np.full(7, 60000.0),
        E2_seg    = np.full(7, 5000.0),
        t_flange  = np.full(4, 2.0),
        E1_flange = np.full(4, 60000.0),
        G_flange  = np.full(4, 5000.0),
        enable_logging = True,
    )

    gd = calc.run()

    # Basic sanity checks
    assert isinstance(gd, GeomData), "run() must return GeomData"
    assert gd.chord == 1500.0

    # Centroid should be within the airfoil bounding box
    assert 0.0 < gd.Xc < 1500.0, f"Xc={gd.Xc} out of range"

    # Inertias must be positive
    assert gd.I_XX > 0, f"I_XX={gd.I_XX} must be positive"
    assert gd.I_ZZ > 0, f"I_ZZ={gd.I_ZZ} must be positive"
    assert gd.I_1 > 0, f"I_1={gd.I_1} must be positive"
    assert gd.I_2 > 0, f"I_2={gd.I_2} must be positive"

    # Cell areas must be positive
    for i in range(3):
        assert gd.A_cells[i] > 0, f"A_cells[{i}]={gd.A_cells[i]} must be positive"

    # Torsional constant must be positive
    assert gd.J > 0, f"J={gd.J} must be positive"

    # Boom areas must be positive
    for i in range(4):
        assert gd.boom_A[i] > 0, f"boom_A[{i}]={gd.boom_A[i]} must be positive"

    # Segment lengths must be positive
    for i in range(7):
        assert gd.s_k[i] > 0, f"s_k[{i}]={gd.s_k[i]} must be positive"

    # 7 T1 segments
    assert len(gd.T1) == 7

    # Equivalent moduli must be positive
    assert gd.E1_eq > 0
    assert gd.G_eq > 0

    # Print summary
    print("\n=== GeomPropCalculator Smoke Test ===")
    print(f"Station Y = {gd.Y_sta} mm")
    print(f"Centroid: ({gd.Xc}, {gd.Zc}) mm")
    print(f"I_XX = {gd.I_XX:.2f} mm^4")
    print(f"I_ZZ = {gd.I_ZZ:.2f} mm^4")
    print(f"I_XZ = {gd.I_XZ:.2f} mm^4")
    print(f"I_1  = {gd.I_1:.2f}, I_2 = {gd.I_2:.2f} mm^4")
    print(f"theta_P = {np.degrees(gd.theta_P):.4f} deg")
    print(f"A_cells = {gd.A_cells}")
    print(f"J = {gd.J:.2f} mm^4")
    print(f"G_REF = {gd.G_REF:.2f} MPa")
    print(f"s_k = {gd.s_k}")
    print(f"boom_A = {gd.boom_A}")
    print(f"Shear center (u,w) = ({gd.us}, {gd.ws})")
    print(f"E1_eq={gd.E1_eq}, E2_eq={gd.E2_eq}, G_eq={gd.G_eq}")
    print("=== PASSED ===\n")


if __name__ == '__main__':
    test_smoke()
