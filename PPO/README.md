PPO-based Local Reactive Controller for Path Tracking

This repository implements a local reactive controller for autonomous path tracking using Proximal Policy Optimization (PPO). The agent is guided by a global path generated via cubic spline interpolation, which provides a smooth trajectory for the agent to follow.

1. Actor-Critic Architecture (PPO)

The learning agent utilizes two specialized neural networks sharing a common backbone: two hidden layers with 64 units each and ReLU activation functions.

Policy Network (The Actor)

Architecture: Input ($S_{dim}=14$) $\rightarrow$ Linear(64) $\rightarrow$ ReLU $\rightarrow$ Linear(64) $\rightarrow$ ReLU $\rightarrow$ Diagonal Gaussian Head.

Action Space: Continuous control with $a_{dim}=1$.

Output: The network predicts a mean ($\mu$) and utilizes a fixed standard deviation ($\sigma$) to define a Normal distribution for action sampling.

PPO Clipping: Implements a clipped objective with a default $\epsilon = 0.2$ to ensure stable updates by limiting the change ratio between the new and old policy.

Value Network (The Critic)

Architecture: Input ($S_{dim}=14$) $\rightarrow$ Linear(64) $\rightarrow$ ReLU $\rightarrow$ Linear(64) $\rightarrow$ ReLU $\rightarrow$ Linear(1).

Purpose: Predicts the expected return $V(s)$.

Advantage Estimation: Uses Generalized Advantage Estimation (GAE) with $\gamma=0.99$ and $\lambda=0.95$ to calculate high-quality advantage targets for the Actor.

2. State Design ($S_{dim} = 14$)

The state vector provides the agent with a "memory" of past movements and a "view" of the upcoming trajectory:

Past Context (6 dimensions): The last two recorded poses $(x, y, yaw)$ from the simulator history.

Future Path (8 dimensions): Four look-ahead waypoints $(x, y)$ sampled at fixed intervals from the cubic spline path relative to the current nearest point.

Normalization: Spatial coordinates are normalized by the environment width (600.0), and $yaw$ is converted to radians to maintain a consistent input scale.

3. Reward Function Design

The reward signal is a composite of distance error, heading alignment, and progress along the path:

$$Reward = 0.8 \cdot e^{-0.1 \cdot d_{min}} + 0.2 \cdot e^{-0.1 \cdot \Delta_{yaw}^2} + R_{progress}$$

Component

Logic

Purpose

Distance Error ($d_{min}$)

Exponential decay based on distance to nearest path point.

Encourages precise path following.

Heading Error ($\Delta_{yaw}$)

Exponential decay based on difference between vehicle and path yaw.

Encourages alignment with path direction.

Progress ($R_{progress}$)

$+0.1$ if moving forward along index; $-1.0$ if moving backward.

Penalizes reversing or standing still.

4. Path Planning & Environment

Trajectory Generation: Uses a 2D Cubic Spline to generate smooth, continuous $(x, y, yaw, curvature)$ trajectories from random anchor points.

Simulation Models: Supports "Basic", "Differential Drive", or "Bicycle" kinematic models.

Multi-Processing: Training is accelerated using a MultiEnv wrapper that runs 8 environments in parallel via Python subprocesses and Pipes.