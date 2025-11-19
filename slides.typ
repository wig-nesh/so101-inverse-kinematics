#import "@preview/diatypst:0.8.0": *
#show: slides.with(
  title: "Under-Actuated Manipulator Control (is hard)",
  subtitle: "& other revelations in manipulator kinematics",
  date: "20.11.2025",
  authors: ("Vignesh Vembar", "Krish Pandya"),
  // footer-title: "",
  // footer-subtitle: "",
  layout: "large",
  count: "number"
)

= Representing the Arm

== DH Parameters

Denavit Hartenberg (DH) parameters describe each joint using four values $a_i, alpha_i, d_i, theta_i$ and produce forward kinematics through a fixed transform pattern. They were popular because they give a compact, minimal representation.

#align(center)[
#image("images/dh_parameters.png", width: 40%)
] 

== Issues with DH Parameters

- Frame assignment rules are unintuitive and heavily constrained.
- Small mistakes in axis alignment break the model.
- Hard to insert tool frames, offsets, or calibration transforms.

#link("https://www.researchgate.net/publication/382909728_From_Unified_Robot_Description_Format_to_DH_Parameters_Examinations_of_Two_Different_Approaches_for_Manipulator")[Some people]
 have tried to bridge the gap between modern methods like URDF and DH, but for this project we used a different approach.

== ETS 

- Elementary Transform Sequence (ETS) represents each joint as a separate transform.
- More intuitive frame assignment.
- Simply compose transforms to get forward kinematics.
- DH parameters might be only 4 parameters per joint, but when ETS has only 6 parameters per joint and is easier to work with, the tradeoff is not worth it.

$
  ""^0 T_e = product_"i=1"^n E_i(eta_i)
$

#quote(attribution: [arxiv.org/abs/2207.01796])[
  This tutorial [...] deliberately avoids the commonly-used Denavit-Hartenberg parameters which we feel confound pedagogy.
]

= Metrics to Visualize Reachability

== Manipulability

$
  omega = sqrt(det(J * J^perp))
$

But for our 5-DOF arm, `J.shape() == (6, 5)`, so $J * J^perp$ is not full rank and $det(J * J^perp) = 0$ always.

So instead, we use the singular values of `J` to define manipulability as:

$
  omega = sqrt(product_"i=1"^5 sigma_i)
$

== Condition

Another useful metric is the condition number of the Jacobian, defined as the ratio of the maximum and minimum singular values:

$
  kappa = sigma_"max" / sigma_"min"
$

$kappa >> 1$ implies the arm is close to a singularity, while $kappa approx 1$ implies good dexterity.

== Visualizing the Reachable Workspace

#align(center)[
  #image("images/reachable_workspace.png")
  #image("images/manipulability_scatter.png", width: 70%)
]

= The Problem and its Naive Solution

== 5-DOF, What are we missing?

#align(center)[
  #image("images/so101.png", width: 60%)
]

There is no end-effector yaw.

== Feasibility-Projected IK

A 5-DOF arm moves on a 5D manifold in 6D task space. We project the desired pose onto this feasible manifold and solve for the corresponding joint update.

*Projection step:*
The achievable motion is the projection of the desired motion onto the range of the Jacobian:
$
Delta x_"achievable" = J J^+ Delta x
$

*Joint update:*
We seek $Delta q$ satisfying $J Delta q approx Delta x_"achievable"$ via damped least-squares:
$
Delta q = arg min_(Delta q) || W_x (J Delta q - Delta x_"achievable") ||^2 + lambda^2 ||Delta q||^2
$

*Derivation:*
Expanding the objective:
$
L(Delta q) = (J Delta q - Delta x_"achievable")^top W_x^top W_x (J Delta q - Delta x_"achievable") + lambda^2 Delta q^top Delta q
$

Taking the derivative with respect to $Delta q$ and setting to zero:
$
(diff L) / (diff Delta q) = 2J^top W_x^top W_x (J Delta q - Delta x_"achievable") + 2lambda^2 Delta q = 0
$

Rearranging:
$
J^top W_x^top W_x J Delta q + lambda^2 Delta q = J^top W_x^top W_x Delta x_"achievable"
$

*Solution:*
Factoring and solving for $Delta q$:
$
(J^top W_x^top W_x J + lambda^2 I) Delta q = J^top W_x^top W_x Delta x_"achievable"
$

Substituting $Delta x_"achievable" = J J^+ Delta x$:
$
Delta q = (J^top W_x^top W_x J + lambda^2 I)^(-1) J^top W_x^top W_x J J^+ Delta x
$

This directly computes joint motions that respect the kinematic constraints while minimizing weighted task-space error.

== Where it Fails

- Because the yaw axis is missing, when the end effector moves off the 2D plane defined by the first joint axis, the IK solver continuously fails to find a solution.
- The "projected" solution is always just on (or very close to) the 2D plane.

= The Workaround that Works

== Fool the IK solver

#align(center)[
  #image("images/so101_side.png", width: 70%)
]

The solver never gets end effector y-axis position or yaw updates, and never updates the first joint angle.

This way the solver doesnt continuously fail when we leave the 2D plane.

== Handling the y-axis position

#align(center)[
  #image("images/drawing.png", width: 40%)
]

$
  theta_"update" &= arctan(y_d,x_d) - arctan(y,x) \
  x_"update" &= sqrt(y'^2+x'^2) - sqrt(y^2+x^2)
$

== Limitations

- Only works for position control, not full 6D pose control.
- End effector yaw is completely uncontrollable, and pitch and roll are constrained to the rotating 2D plane.


= Demo