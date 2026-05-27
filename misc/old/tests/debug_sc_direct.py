'''
Compute shear center using direct Megson method (independent validation).

For unit vertical shear S_Z = 1 at the SC, zero twist:
1. Compute open-section flows q_b via chain analysis
2. Solve redundant flows q_s0 from rate-of-twist compatibility
3. Total flow = q_b + q_s0
4. Shear center X from moment equilibrium about P1
'''
import sys
from pathlib import Path
import json
import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))

from geometry.geom_properties import GeomPropCalculator

path = _ROOT / "data" / "airfoils" / "wortmannfx63137_AirfoilData.json"
with open(path) as f:
    d = json.load(f)
afl = {k: np.array(v) for k, v in d.items()}

calc = GeomPropCalculator(
    x_upper=afl['x_upper'], y_upper=afl['y_upper'],
    x_lower=afl['x_lower'], y_lower=afl['y_lower'],
    x_camber=afl['x_camber'], y_camber=afl['y_camber'],
    chord=1500.0, twist=0.0, Y_sta=2000.0,
    xw1=0.25, xw2=0.65,
    bf1=20.0, bf2=20.0, bf3=15.0, bf4=15.0,
    t_seg=np.full(7, 1.0), G_seg=np.full(7, 5000.0),
    E1_seg=np.full(7, 60000.0), E2_seg=np.full(7, 5000.0),
    t_flange=np.full(4, 2.0), E1_flange=np.full(4, 60000.0),
    G_flange=np.full(4, 5000.0), enable_logging=False,
)

calc._scale_airfoil()
calc._find_spar_intersections()
calc._segment_T1()
calc._compute_segment_properties()
calc._compute_delta_matrix()
calc._compute_cell_areas()
calc._compute_Delta_and_J()
calc._compute_centroid()
calc._compute_inertia()
calc._compute_principal_inertia()
calc._compute_boom_areas()

Xc, Zc = calc.Xc, calc.Zc
IXX, IZZ, IXZ = calc.I_XX, calc.I_ZZ, calc.I_XZ
D = IXX * IZZ - IXZ**2
u, w, B = calc.boom_u, calc.boom_w, calc.boom_A
dk = calc.delta_k

print("=" * 60)
print("DIRECT SHEAR CENTER COMPUTATION (Megson method)")
print("=" * 60)

# Step 1: Open-section flows for unit S_Z
# dq_Z = (IXZ*u - IZZ*w) / D * B  (thesis Eq 3.39)
dqZ = (IXZ * u - IZZ * w) / D * B
print(f"dqZ per boom = {dqZ}")
print(f"Sum dqZ = {np.sum(dqZ):.6e}")

# Chain: B1->seg6->B2->seg5(B2->B4)->B4->seg7(B4->B3)->B3
# Flow in each segment (in the SEGMENT direction):
q_seg = np.zeros(7)  # q_b for each T1 segment

# seg1-4: cut or zero
q_seg[0] = 0.0  # seg1 (cut)
q_seg[1] = 0.0  # seg2 (cut)
q_seg[2] = 0.0  # seg3 (cut)
q_seg[3] = 0.0  # seg4 (zero by extension)

# seg6 (idx 5): B1->B2, flow = dqZ[0]
q_seg[5] = dqZ[0]

# seg5 (idx 4): defined B4->B2, actual chain flow goes B2->B4
# Flow in seg5 direction (B4->B2) = -(dqZ[0]+dqZ[1])
q_seg[4] = -(dqZ[0] + dqZ[1])

# seg7 (idx 6): defined B3->B4, chain flow B4->B3
# Flow in seg7 direction (B3->B4) = -(dqZ[0]+dqZ[1]+dqZ[3])
q_seg[6] = -(dqZ[0] + dqZ[1] + dqZ[3])

print(f"\nOpen-section flows q_b (in segment direction):")
for i in range(7):
    label = calc.T1[i]['label']
    print(f"  {label}: q_b = {q_seg[i]:.6e}")

# Verify: q_b,34 = dqZ[2] (= -(dq1+dq2+dq4) by equilibrium)
print(f"\nVerify: dqZ[2]={dqZ[2]:.6e}, -(dq1+dq2+dq4)={-(dqZ[0]+dqZ[1]+dqZ[3]):.6e}")

# Step 2: O vectors (integral of q_b * delta around each cell)
# Cell circulation: clockwise from behind (-Y)
# Cell I clockwise: B1->seg1_rev->B2->seg6_rev->B1
#   seg1_rev: q_b=0, seg6_rev: flow in B2->B1 = -q_seg[5]
#   O_I = (-q_seg[5]) * dk[5]
O_I = -q_seg[5] * dk[5]

# Cell II clockwise: B1->seg6->B2->seg5_rev(B2->B4)->B4->seg7_rev(B4->B3)->B3->seg2_rev->B1
#   seg6: flow B1->B2 (same dir) -> +q_seg[5] * dk[5]
#   seg5_rev (B2->B4): flow in B2->B4 = -q_seg[4] (since q_seg[4] is B4->B2)
#   -> (-q_seg[4]) * dk[4]
#   seg7_rev (B4->B3): flow in B4->B3 = -q_seg[6] (since q_seg[6] is B3->B4)
#   -> (-q_seg[6]) * dk[6]
#   seg2_rev: q_b=0
O_II = q_seg[5]*dk[5] + (-q_seg[4])*dk[4] + (-q_seg[6])*dk[6]

# Cell III clockwise: B3->seg7->B4->seg4_rev->P4->seg3_rev->B3
#   seg7: flow B3->B4 (same dir) -> +q_seg[6] * dk[6]
#   seg4_rev, seg3_rev: q_b=0
O_III = q_seg[6] * dk[6]

O_vec = np.array([O_I, O_II, O_III])
print(f"\nO vector = {O_vec}")

# Step 3: Solve compatibility: [delta]{q_s0} = -{O}
delta_inv = np.linalg.inv(calc.delta_mat)
qs0 = -delta_inv @ O_vec
print(f"q_s0 = {qs0}")

# Step 4: Total flows in each segment (in segment direction)
# Add redundant flows according to cell membership
# q_s0 positive = clockwise for each cell

q_total = np.zeros(7)

# seg1 (cell I): cell I clockwise traverses seg1 as B1->B2 (reversed from seg def B2->B1)
# In seg1 direction (B2->B1): -q_s0[0]
q_total[0] = q_seg[0] + (-qs0[0])

# seg2 (cell II): cell II clockwise traverses seg2 reversed (B3->B1)
# In seg2 direction (B1->B3): -q_s0[1]
q_total[1] = q_seg[1] + (-qs0[1])

# seg3 (cell III): cell III clockwise traverses seg3 reversed (P4->B3)
# In seg3 direction (B3->P4): -q_s0[2]
q_total[2] = q_seg[2] + (-qs0[2])

# seg4 (cell III): cell III clockwise traverses seg4 reversed (B4->P4)
# In seg4 direction (P4->B4): -q_s0[2]
q_total[3] = q_seg[3] + (-qs0[2])

# seg5 (cell II only): cell II clockwise traverses B2->B4 = reversed from seg5 B4->B2
# In seg5 direction (B4->B2): -q_s0[1]
q_total[4] = q_seg[4] + (-qs0[1])

# seg6 (shared I and II):
# Cell I clockwise at seg6: B2->B1 (reversed from seg6 B1->B2) -> contributes -q_s0[0]
# Cell II clockwise at seg6: B1->B2 (same as seg6) -> contributes +q_s0[1]
# In seg6 direction (B1->B2): +q_s0[1] - q_s0[0]
q_total[5] = q_seg[5] + (qs0[1] - qs0[0])

# seg7 (shared II and III):
# Cell II clockwise at seg7: B4->B3 (reversed from seg7 B3->B4) -> contributes -q_s0[1]
# Cell III clockwise at seg7: B3->B4 (same as seg7) -> contributes +q_s0[2]
# In seg7 direction (B3->B4): +q_s0[2] - q_s0[1]
q_total[6] = q_seg[6] + (qs0[2] - qs0[1])

print(f"\nTotal flows (in segment direction):")
for i in range(7):
    print(f"  {calc.T1[i]['label']}: q_total = {q_total[i]:.6e}")

# Step 5: Moment about P1 for shear center X
# M = S_Z * eta_s = sum(q_i * swept_i)
# where swept_i is computed in the segment direction
M_total = 0.0
for i in range(7):
    seg = calc.T1[i]
    x = seg['x'] - calc.P1[0]
    z = seg['z'] - calc.P1[1]
    swept_i = float(np.sum(x[:-1]*z[1:] - x[1:]*z[:-1]))
    M_i = q_total[i] * swept_i
    M_total += M_i
    if abs(q_total[i]) > 1e-10:
        print(f"  M_{calc.T1[i]['label']} = {M_i:.2f} (q={q_total[i]:.6e}, swept={swept_i:.1f})")

# For S_Z = 1: eta_s = M_total
eta_s = M_total
print(f"\neta_s (X of SC from BA) = {eta_s:.2f} mm")
print(f"eta_s as %chord = {eta_s/1500*100:.1f}%")

# Now do the same for S_X = 1 to get xi_s
dqX = (-IXX * u + IXZ * w) / D * B
q_segX = np.zeros(7)
q_segX[5] = dqX[0]
q_segX[4] = -(dqX[0] + dqX[1])
q_segX[6] = -(dqX[0] + dqX[1] + dqX[3])

O_IX = -q_segX[5]*dk[5]
O_IIX = q_segX[5]*dk[5] + (-q_segX[4])*dk[4] + (-q_segX[6])*dk[6]
O_IIIX = q_segX[6]*dk[6]
O_vecX = np.array([O_IX, O_IIX, O_IIIX])

qs0X = -delta_inv @ O_vecX

q_totalX = np.zeros(7)
q_totalX[0] = -qs0X[0]
q_totalX[1] = -qs0X[1]
q_totalX[2] = -qs0X[2]
q_totalX[3] = -qs0X[2]
q_totalX[4] = q_segX[4] + (-qs0X[1])
q_totalX[5] = q_segX[5] + (qs0X[1] - qs0X[0])
q_totalX[6] = q_segX[6] + (qs0X[2] - qs0X[1])

M_totalX = 0.0
for i in range(7):
    seg = calc.T1[i]
    x = seg['x'] - calc.P1[0]
    z = seg['z'] - calc.P1[1]
    swept_i = float(np.sum(x[:-1]*z[1:] - x[1:]*z[:-1]))
    M_totalX += q_totalX[i] * swept_i

# For S_X = 1: xi_s from moment equilibrium
# M = S_X * xi_s  ... but sign: for horizontal force S_X applied at (eta_s, xi_s),
# the moment about P1 = S_X * (xi_s_P1 - xi_s_SC) ... hmm
# Actually: S_X * w_arm = sum(q * swept)
# where w_arm is the Z-coordinate of SC from P1 (= xi_s)
# But the sign depends on the moment convention.
# Let's compute it: if we apply S_X at the SC, the moment about P1 in the positive
# (counterclockwise) direction is S_X * (z_SC - z_P1) = S_X * xi_s
# And the shear flows generate: sum(q * swept/2) where swept is the signed area
xi_s = M_totalX
print(f"\nxi_s (Z of SC from BA) = {xi_s:.2f} mm")
print(f"xi_s as %chord = {xi_s/1500*100:.1f}%")

print(f"\nSC global: ({eta_s:.2f}, {xi_s:.2f})")
print(f"SC centroidal: ({eta_s - Xc:.2f}, {xi_s - Zc:.2f})")
print(f"\nExpected: eta_s ~ 375-525mm, xi_s ~ 0-66mm")
