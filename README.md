# 2-DOF Planar Manipulator — Dynamics & Control

MuJoCo simulation of a 2-DOF revolute-revolute manipulator. Dynamics are computed analytically in Python using the **Uicker–Kahn Lagrangian formulation**, and control is implemented as **Computed Torque Control (CTC)**.

---

## Table of Contents

1. [System Description](#1-system-description)
2. [Homogeneous Transformations](#2-homogeneous-transformations)
3. [The Q Matrix (Joint Velocity Operator)](#3-the-q-matrix-joint-velocity-operator)
4. [Uicker–Kahn U-Matrices](#4-uickerkahn-u-matrices)
5. [Pseudo-Inertia Matrix](#5-pseudo-inertia-matrix)
6. [Equations of Motion (Lagrangian Derivation)](#6-equations-of-motion-lagrangian-derivation)
   - [Inertia Matrix D(q)](#61-inertia-matrix-dq)
   - [Coriolis & Centrifugal Vector H(q, q̇)](#62-coriolis--centrifugal-vector-hq-q)
   - [Gravity Vector G(q)](#63-gravity-vector-gq)
7. [Full Equations of Motion](#7-full-equations-of-motion)
8. [Computed Torque Control (CTC)](#8-computed-torque-control-ctc)
   - [Control Law](#81-control-law)
   - [Closed-Loop Error Dynamics](#82-closed-loop-error-dynamics)
   - [Stability Analysis](#83-stability-analysis)
9. [PD Gain Tuning](#9-pd-gain-tuning)
10. [Implementation Notes](#10-implementation-notes)
11. [References](#11-references)

---

## 1. System Description

A 2-DOF RR (revolute–revolute) planar manipulator with both joints rotating about the Y-axis (vertical plane, gravity acting in −Z).

| Parameter | Symbol | Value |
|-----------|--------|-------|
| Link 1 length | L₁ | 1.0 m |
| Link 2 length | L₂ | 1.0 m |
| Link 1 mass | m₁ | 1.0 kg |
| Link 2 mass | m₂ | 1.0 kg |
| CoM position on link i | rᵢ | 0.5 m (midpoint) |
| Gravity | g | 9.81 m/s² |

**Generalized coordinates:**

```
q = [q₁, q₂]ᵀ ∈ ℝ²
```

**Trajectory tracked:**

```
q_ref(t) = [0,  (π/4)·cos(t)]ᵀ
q̇_ref(t) = [0, -(π/4)·sin(t)]ᵀ
q̈_ref(t) = [0, -(π/4)·cos(t)]ᵀ
```

---

## 2. Homogeneous Transformations

Each link's transform is a **rotation about Y** followed by a **translation along the rotated X-axis** (standard DH convention for a revolute joint):

```
            ⎡  cos(q)   0   sin(q)   L·cos(q) ⎤
T_y(q, L) = ⎢    0      1     0        0      ⎥
            ⎢ -sin(q)   0   cos(q)  -L·sin(q) ⎥
            ⎣    0      0     0        1       ⎦
```

This is equivalent to:

```
T_y(q, L) = Rot_y(q) · Trans_x(L)
```

where `Trans_x(L)` translates by L along the local X-axis.

The cumulative transforms from the world frame are:

```
T₁ = T_y(q₁, L₁)
T₂ = T_y(q₁, L₁) · T_y(q₂, L₂)
```

So `Tᵢ = T_{0→i}` maps the i-th link's tip frame to the world frame.

---

## 3. The Q Matrix (Joint Velocity Operator)

For a **revolute joint rotating about Y**, the Q matrix encodes `∂R_y(q)/∂q · R_y(q)⁻¹` in homogeneous form:

```
    ⎡  0   0   1   0 ⎤
Q = ⎢  0   0   0   0 ⎥
    ⎢ -1   0   0   0 ⎥
    ⎣  0   0   0   0 ⎦
```

**Physical meaning:** For a revolute joint with transform T(q),

```
∂T/∂q = Q · T
```

So Q is the **infinitesimal generator** of rotation about Y, lifted to SE(3). It satisfies:

```
Q² = -I₃ (on the rotational block)  →  analogous to the cross-product matrix for angular velocity
```

For a joint rotating about axis **ω̂**, the general Q is:

```
        ⎡  [ω̂]×   0 ⎤
Q_ω =   ⎢           ⎥
        ⎣   0ᵀ     0 ⎦
```

where `[ω̂]×` is the skew-symmetric matrix of ω̂. For Y-axis: `[ê_y]× = [[0,0,1],[0,0,0],[-1,0,0]]`.

---

## 4. Uicker–Kahn U-Matrices

The **U-matrix** is the partial derivative of the transform `T_{0→i}` with respect to joint variable `qⱼ`:

```
U_{j,i} = ∂T_{0→i} / ∂qⱼ
```

**Derivation** (chain rule on the product of transforms):

```
T_{0→i} = T₁ · T₂ · ... · Tᵢ

∂T_{0→i}/∂qⱼ = T_{0→j-1} · (∂Tⱼ/∂qⱼ) · T_{j→i}
              = T_{0→j-1} · (Q · Tⱼ) · T_{j→i}
              = T_{0→j-1} · Q · T_{j-1→i}
```

Since `Tⱼ = T_{0→j-1}⁻¹ · T_{0→j}` and using `T_{j-1→i} = T_{0→j-1}⁻¹ · T_{0→i}`:

```
          ⎧ T_{0→j-1} · Q · T_{0→j-1}⁻¹ · T_{0→i}    if j ≤ i
U_{j,i} = ⎨
          ⎩ 0                                          if j > i
```

This is exactly what `get_U(T, Q, j, i)` computes:

```python
U[j,i] = T[j-1] @ Q @ inv(T[j-1]) @ T[i]
```

**Second-order U-matrix** (for Coriolis/centrifugal):

```
U_{jk,i} = ∂²T_{0→i} / (∂qⱼ ∂qₖ)
```

For j ≤ k ≤ i:

```
U_{jk,i} = T_{0→j-1} · Q · T_{j-1→k-1} · Q · T_{k-1→i}
```

which `get_U_dot(T, Q, j, k, i)` computes as:

```python
U[j,k,i] = T[m1-1] @ Q @ (inv(T[m1-1]) @ T[m2-1]) @ Q @ (inv(T[m2-1]) @ T[i])
```

where m1 = min(j,k), m2 = max(j,k) (since partial derivatives commute).

---

## 5. Pseudo-Inertia Matrix

The **pseudo-inertia matrix** Jᵢ is a 4×4 symmetric matrix encoding the rigid body inertia of link i:

```
     ⎡ (Iyy+Izz-Ixx)/2    -Ixy          -Ixz        m·cx ⎤
     ⎢     -Ixy       (Ixx+Izz-Iyy)/2  -Iyz        m·cy ⎥
Jᵢ = ⎢     -Ixz           -Iyz      (Ixx+Iyy-Izz)/2  m·cz ⎥
     ⎣    m·cx            m·cy           m·cz          m  ⎦
```

where `(cx, cy, cz)` is the CoM position in the link frame and `Ixx, Iyy, Izz, ...` are the inertia tensor components.

**Simplified form used here** (point mass at CoM on X-axis, no products of inertia):

Assuming the CoM is at `(rᵢ, 0, 0)` in the link frame and treating each link as a slender rod:

```
Jᵢ = diag([m·r², 0, m·r², m])
```

This corresponds to:
- `Ixx = 0` (rotation about X — along the rod — is zero for a point mass)
- `Iyy = Izz = m·r²` (rotation about Y and Z)
- CoM at origin of the pseudo-inertia frame

> **Note:** The `diaginertia` in the MuJoCo XML (`0.1 0.1 0.01`) reflects a more realistic capsule geometry. The Python controller uses a simplified point-mass approximation — a model–controller mismatch that the robustness of CTC handles well for small perturbations.

---

## 6. Equations of Motion (Lagrangian Derivation)

The standard manipulator dynamics derived via the Lagrangian are:

```
D(q)·q̈ + H(q, q̇) + G(q) = τ
```

The Uicker–Kahn formulation expresses all three terms via traces of matrix products, enabling a unified and algorithmically consistent derivation.

### 6.1 Inertia Matrix D(q)

```
D_{jk} = Σᵢ₌ₘₐₓ(j,k)ⁿ  Tr( U_{k,i} · Jᵢ · U_{j,i}ᵀ )
```

- **Symmetric:** D = Dᵀ (from symmetry of the trace and Jᵢ)
- **Positive definite:** D ≻ 0 for non-degenerate configurations
- **Dimensions:** D ∈ ℝ²ˣ²

The trace formulation arises from the kinetic energy:

```
T = (1/2) · q̇ᵀ · D(q) · q̇
  = (1/2) · Σᵢ Tr( (Σⱼ U_{j,i}·q̇ⱼ) · Jᵢ · (Σₖ U_{k,i}·q̇ₖ)ᵀ )
```

Expanding and extracting the coefficient of `q̇ⱼ·q̇ₖ` gives the formula above.

### 6.2 Coriolis & Centrifugal Vector H(q, q̇)

```
Hⱼ = Σᵢ Σₖ Σₘ  Tr( U_{km,i} · Jᵢ · U_{j,i}ᵀ ) · q̇ₖ · q̇ₘ
```

This is the full Coriolis + centrifugal term. It arises from:

```
H = Ċ(q, q̇) · q̇  where  Cⱼₖ = (1/2) Σₘ [Γⱼₖₘ + Γⱼₘₖ - Γₖₘⱼ] · q̇ₘ
```

with Christoffel symbols `Γⱼₖₘ = ∂D_{jk}/∂qₘ`, but the U-matrix form computes this directly without explicitly forming Christoffel symbols.

**Note:** In the code, `H` is a vector (not matrix) — it is `C(q,q̇)·q̇` already evaluated, not the Coriolis matrix itself.

### 6.3 Gravity Vector G(q)

```
Gⱼ = -Σᵢ mᵢ · g⃗ᵀ · U_{j,i} · r̄ᵢ
```

where:
- `g⃗ = [0, -g, 0, 0]ᵀ` — gravity in world frame (4D homogeneous)
- `r̄ᵢ = [rᵢ, 0, 0, 1]ᵀ` — CoM position in link-i frame (homogeneous)

Derivation from potential energy:

```
Vᵢ = mᵢ · g⃗ᵀ · T_{0→i} · r̄ᵢ
Gⱼ = ∂V/∂qⱼ = Σᵢ mᵢ · g⃗ᵀ · (∂T_{0→i}/∂qⱼ) · r̄ᵢ
              = Σᵢ mᵢ · g⃗ᵀ · U_{j,i} · r̄ᵢ
```

(with sign negated because `G` is moved to the left side: `D·q̈ + H + G = τ`)

---

## 7. Full Equations of Motion

```
D(q)·q̈ + H(q, q̇) + G(q) = τ
```

Expanded for the 2-DOF case:

```
⎡ D₁₁  D₁₂ ⎤ ⎡ q̈₁ ⎤   ⎡ H₁ ⎤   ⎡ G₁ ⎤   ⎡ τ₁ ⎤
⎢          ⎥ ⎢    ⎥ + ⎢    ⎥ + ⎢    ⎥ = ⎢    ⎥
⎣ D₂₁  D₂₂ ⎦ ⎣ q̈₂ ⎦   ⎣ H₂ ⎦   ⎣ G₂ ⎦   ⎣ τ₂ ⎦
```

Key properties:
- **D is configuration-dependent:** D = D(q)
- **Skew-symmetry property:** `Ṁ - 2C` is skew-symmetric (useful for passivity-based control)
- **Bounded inertia:** `λ_min·I ≼ D(q) ≼ λ_max·I` for bounded joint angles

---

## 8. Computed Torque Control (CTC)

Also known as **Inverse Dynamics Control** or **Feedback Linearization** for manipulators.

### 8.1 Control Law

Define the virtual control input:

```
u = q̈_ref + Kp·(q_ref - q) + Kd·(q̇_ref - q̇)
```

where `Kp, Kd ∈ ℝ²ˣ²` are diagonal gain matrices.

Apply the full inverse dynamics to cancel the nonlinear terms:

```
τ = D(q)·u + H(q, q̇) + G(q)
```

Substituting the actual dynamics `D·q̈ + H + G = τ`:

```
D·q̈ = D·u     →     q̈ = u
```

The nonlinear dynamics are exactly cancelled (assuming perfect model knowledge), reducing the system to:

```
q̈ = q̈_ref + Kp·e + Kd·ė
```

### 8.2 Closed-Loop Error Dynamics

Define the tracking error:

```
e = q_ref - q
ė = q̇_ref - q̇
ë = q̈_ref - q̈
```

Substituting `q̈ = u`:

```
q̈_ref - ë = q̈_ref + Kp·e + Kd·ė
          ↓
ë + Kd·ė + Kp·e = 0
```

This is a **decoupled, linear, second-order** error system — one per joint:

```
ë_i + Kd_i·ė_i + Kp_i·e_i = 0,    i = 1, 2
```

Characteristic polynomial per joint:

```
s² + Kd_i·s + Kp_i = 0
```

### 8.3 Stability Analysis

The error dynamics are globally asymptotically stable (GAS) if and only if:

```
Kp_i > 0    and    Kd_i > 0
```

(standard Routh–Hurwitz for a 2nd-order system with positive coefficients)

Using a Lyapunov function for the error system:

```
V(e, ė) = (1/2)·ėᵀ·ė + (1/2)·eᵀ·Kp·e > 0
V̇ = ėᵀ·ë + eᵀ·Kp·ė
  = ėᵀ·(-Kd·ė - Kp·e) + eᵀ·Kp·ė
  = -ėᵀ·Kd·ė ≤ 0
```

V̇ = 0 only when `ė = 0`, and by LaSalle's invariance principle, the system converges to `e = 0, ė = 0`.

> **Important:** This GAS result holds only under perfect model knowledge. With model errors `ΔD, ΔH, ΔG`, the system becomes:
> ```
> ë + Kd·ė + Kp·e = D⁻¹·(ΔD·q̈ + ΔH + ΔG)
> ```
> — a bounded disturbance input. Stability is preserved if gains dominate the disturbance, but robustness guarantees require robust or adaptive extensions.

---

## 9. PD Gain Tuning

Gains are chosen to achieve a desired second-order response. Given:

```
s² + Kd·s + Kp = 0   ↔   s² + 2ζωₙ·s + ωₙ² = 0
```

Mapping:

```
Kp = ωₙ²
Kd = 2ζωₙ
```

For `Kp = 200, Kd = 20`:

```
ωₙ = √200 ≈ 14.1 rad/s
ζ  = 20 / (2·14.1) ≈ 0.71   (slightly underdamped, near critical)
```

Settling time (2% criterion):

```
tₛ ≈ 4 / (ζ·ωₙ) = 4 / 10 = 0.4 s
```

Overshoot:

```
%OS = exp(-πζ / √(1-ζ²)) × 100 ≈ 4.3%
```

---

## 10. Implementation Notes

| Aspect | Detail |
|--------|--------|
| **Simulation timestep** | 2 ms (`timestep="0.002"`) — well within Nyquist for ωₙ ≈ 14 rad/s |
| **Discretization** | Continuous-time CTC applied at every step (no explicit discretization of controller) |
| **Model mismatch** | MuJoCo uses capsule geometry; Python uses point-mass inertia → slight mismatch absorbed by PD gains |
| **Gravity convention** | MuJoCo: `gravity="0 0 -9.81"` (−Z); Python: `g_vec = [0, -g, 0, 0]` in Y-axis DH frame → joints rotate in XZ plane under gravity acting in −Z, projected through the −Y component of the Y-rotation convention |
| **Actuator model** | Ideal torque motors (`<motor gear="1"/>`) — no actuator dynamics |
| **Anti-windup** | Not needed — CTC is a feedforward-dominated scheme with no integrator |
| **Computational cost** | O(n³) in number of joints — efficient for small n; use recursive Newton–Euler (O(n)) for n > 6 |

---

## 11. References

1. **Uicker, J.J. (1965)** — "On the Dynamic Analysis of Spatial Linkages Using 4×4 Matrices." Original formulation of the U-matrix method.

2. **Paul, R.P. (1981)** — *Robot Manipulators: Mathematics, Programming, and Control.* MIT Press. — Standard reference for homogeneous transform dynamics.

3. **Siciliano, B. et al. (2009)** — *Robotics: Modelling, Planning and Control.* Springer. — Chapter 7 (Dynamics), Chapter 8 (Motion Control). Primary reference for CTC and Lagrangian formulation.

4. **Slotine, J.J.E. & Li, W. (1991)** — *Applied Nonlinear Control.* Prentice Hall. — Chapter 9 for stability analysis of CTC and passivity properties.

5. **Spong, M.W., Hutchinson, S. & Vidyasagar, M. (2006)** — *Robot Modeling and Control.* Wiley. — Chapter 6–7 for pseudo-inertia matrix and Lagrangian dynamics.
