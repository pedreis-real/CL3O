'''Debug shear center with all fixes applied.'''
import sys
from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))

from geometry.geom_properties import GeomPropCalculator

path = _ROOT / "data" / "airfoils" / "e169_AirfoilData.json"
with open(path) as f:
    d = json.load(f)
afl = {k: np.array(v) for k, v in d.items()}

xw1 = 60 / 300
xw2 = 210 / 300
chord = 300

E1 = 200000.0
E2 = 10000.0
G  = 6000.0
t_seg = np.asarray([1.8, 1.8, 1.8, 1.8, 1.8, 4.0, 2.0])
t_flange = np.asarray([3.0, 2.5, 3.0, 2.0])

calc = GeomPropCalculator(
    x_upper=afl['x_upper'], y_upper=afl['y_upper'],
    x_lower=afl['x_lower'], y_lower=afl['y_lower'],
    x_camber=afl['x_camber'], y_camber=afl['y_camber'],
    chord=chord, twist=0.0, Y_sta=0,
    xw1=xw1, xw2=xw2,
    bf1=12.0, bf2=10.0, bf3=8.0, bf4=8.0,
    t_seg=t_seg, G_seg=np.full(7, G),
    E1_seg=np.full(7, E1), E2_seg=np.full(7, E2),
    t_flange=t_flange, E1_flange=np.full(4, E1),
    G_flange=np.full(4, G), enable_logging=False,
)

props = calc.run()

'''
Xc, Zc = props.Xc, props.Zc
IXX, IZZ, IXZ = props.I_XX, props.I_ZZ, props.I_XZ
D = IXX * IZZ - IXZ**2
u, w, B = props.boom_u, props.boom_w, props.boom_A
dk = props.delta_k
sk = props.s_k

print(f"Centroid: ({Xc:.2f}, {Zc:.2f})")
print(f"IXX={IXX:.0f}  IZZ={IZZ:.0f}  IXZ={IXZ:.0f}")

# --- Fix 1: Correct dq formulas from thesis Eq 3.38-3.39 ---
# Eq 3.38: dqX = (-IXX*u + IXZ*w) / D * B
# Eq 3.39: dqZ = (+IXZ*u - IZZ*w) / D * B
dqX = (-IXX * u + IXZ * w) / D * B
dqZ = ( IXZ * u - IZZ * w) / D * B

# --- Fix 2: Correct q_b,24 from chain analysis ---
# Chain: B1->seg6->B2->seg5(B2->B4)->B4->seg7(B4->B3)->B3
# q_b,12 = dq1, q_b,34 = dq3, q_b,24 = dq1+dq2
qbX_12 = dqX[0]
qbX_34 = dqX[2]
qbX_24 = (qbX_12 + dqX[1]) - (qbX_34 + dqX[3])

qbZ_12 = dqZ[0]
qbZ_34 = dqZ[2]
qbZ_24 = (qbZ_12 + dqZ[1]) - (qbZ_34 + dqZ[3])

print(f"\ndqX = {dqX}")
print(f"dqZ = {dqZ}")
print(f"qbX: 12={props.qbX_12:.6e}, 34={props.qbX_34:.6e}, 24={props.qbX_24:.6e}")
print(f"qbZ: 12={props.qbZ_12:.6e}, 34={props.qbZ_34:.6e}, 24={props.qbZ_24:.6e}")

# --- O* vectors (Eq 3.42) ---
d5, d6, d7 = dk[4], dk[5], dk[6]
OX = np.array([
    -qbX_12 * d6,
     qbX_12 * d6 + qbX_24 * d5 - qbX_34 * d7,
     qbX_34 * d7,
])
OZ = np.array([
    -qbZ_12 * d6,
     qbZ_12 * d6 + qbZ_24 * d5 - qbZ_34 * d7,
     qbZ_34 * d7,
])
print(f"\nOX = {OX}")
print(f"OZ = {OZ}")

# --- Fix 3: Correct moment computation ---
# swept6 along seg6 direction (B1->B2): used with q_b,12 (same direction) -> M = q*swept
# swept5 along seg5 direction (B4->B2): but q_b,24 goes B2->B4 -> M = q*(-swept5) = -q*swept5
# swept7 along seg7 direction (B3->B4): used with q_b,34 (same direction) -> M = q*swept
A12 = props._swept_double_area(5)  # seg6
A24 = props._swept_double_area(4)  # seg5
A34 = props._swept_double_area(6)  # seg7

print(f"\nswept: seg6={A12:.1f}, seg5={A24:.1f}, seg7={A34:.1f}")
print(f"p0_12*s6={xw1*chord*sk[5]:.1f}, p0_34*s7={xw2*chord*sk[6]:.1f}\n")

# q_b,24 flows B2->B4 but seg5 is B4->B2, so negate swept5
MqbX = -qbX_12 * A12 + qbX_24 * A24 - qbX_34 * A34
MqbZ = -qbZ_12 * A12 + qbZ_24 * A24 - qbZ_34 * A34

print(f"MqbX = {MqbX:.4f}")
print(f"MqbZ = {MqbZ:.4f}")

# --- Solve for shear center ---
delta_inv = np.linalg.inv(props.delta_mat)
A_vec = 2.0 * props.A_cells

A_dinv_OX = float(A_vec @ delta_inv @ OX)
A_dinv_OZ = float(A_vec @ delta_inv @ OZ)

print(f"A_dinv_OX = {A_dinv_OX:.4f}")
print(f"A_dinv_OZ = {A_dinv_OZ:.4f}")

xi_s  = -MqbZ + A_dinv_OZ    # Eq 3.49
eta_s =  MqbX - A_dinv_OX    # Eq 3.47

Xs = -xi_s
Zs = -eta_s

us = Xs - Xc
ws = Zs - Zc

print(f"xi_s  = {xi_s:.2f} mm (X from BA)")
print(f"\neta_s = {eta_s:.2f} mm (Z from BA)")
print(f"SC global: ({Xs:.2f}, {Zs:.2f})")
print(f"SC centroidal: ({us:.2f}, {ws:.2f})")
print(f"SC as %chord: X={Xs/chord*100:.1f}%, Z={Zs/chord*100:.1f}%")

print(f"\nExpected: X_SC_global ~ 375-525mm (25-35% chord)")
print(f"Expected: Z_SC_global ~ 0-66mm (near centroid Z={Zc:.1f})")

# --- Plot ---
fig, ax = plt.subplots(figsize=(12,2))

xu = afl['x_upper'] * chord
yu = afl['y_upper'] * chord
xl = afl['x_lower'] * chord
yl = afl['y_lower'] * chord
xc = afl['x_camber'] * chord
yc = afl['y_camber'] * chord

ax.plot(xu, yu, color="#663363", label=r"Upper")
ax.plot(xl, yl, color="#ca4816", label=r"Lower")
ax.plot(xc, yc, linestyle="-.", color="#2a2b2c", label=r"Camber")

ax.plot([u[1], u[0]]+Xc, [w[1], w[0]]+Zc, color="#48a33b", label=r"Aft Spar")
ax.plot([u[3], u[2]]+Xc, [w[3], w[2]]+Zc, color="#1b6511", label=r"Rear Spar")

ax.scatter(Xc, Zc, linewidth=5, color="#D2B026", label=r"Centroid")
ax.scatter(Xs, Zs, linewidth=5, color="#17B18A", label=r"Shear Center")

ax.grid(True, which='both',alpha=0.6)

ax.set_xlabel(r"X [mm]")
ax.set_ylabel(r"Z [mm]")

ax.legend(
    loc="upper right",
    fontsize=8,
    framealpha=0.92,
    edgecolor="#cccccc",
)

plt.show()'''