---
title: Value Based Reinforce
date: 2025-03-01 14:35:02
tags: [RL, LLM, ShusenWang]
categories: [LLM]
mathjax: true
---

# Value Based Reinforce

## Deep Q Network(DQN)

Use neural network $Q(s,a;\mathbf{w})$ to approximate the Q Function $Q^*(s,a)$. 

## Temporal Difference (TD) Learning

$$
\begin{align}
U_t &= R_t + \gamma R_{t+1} +\gamma^2 R_{t+2} + \gamma^3 R_{t+3}..... \\
    &= R_t + \gamma(R_{t+1} +\gamma R_{t+2} + \gamma^2 R_{t+3}.....) \\
    &= R_t + \gamma U_{t+1}  \\ \\
    
Q(s_t,a_t;\mathbf{w}) & \approx  \mathbb{E}(U_t) \\
				& \approx \mathbb{E}(R_t + \gamma U_{t+1}) \\
				& \approx \mathbb{E}(R_t + \gamma Q(S_{t+1},A_{t+1};\mathbf{w})) \\
                & \approx  r_t + \gamma \cdot Q(s_{t+1},a_{t+1};\mathbf{w})  \\

\end{align}
$$

- Prediction: $Q(s_t,a_t;\mathbf{w}_t)$

- TD target: 

  $y_t = r_t + \gamma Q(s_{t+1},a_{t+1};\mathbf{w}_t) = r_t + \gamma \  \mathop{max}\limits_a Q(s_{t+1},a;\mathbf{w}_t)$

- Loss: $L_t = \frac{1}{2} [Q(s_t,a_t;\mathbf{w}) - y_t]^2$.

- Gradient descent: $\mathbf{w}_{t+1} = \mathbf{w}_t - \alpha \cdot \frac{\partial L_t}{\partial \mathbf{w}} |_\mathbf{w = w_t}$ 

### Algorithm: one iteration of TD learning

1. Observe state $S_t = s_t$ and action $A_t = a_t$
2. Predict the value: $q_t = Q(s_t,a_t;\mathbf{w}_t)$.
3. Differentiate the value network: $\mathbf{d}_t = \frac{\partial Q(s_t,a_t;\mathbf{w})}{\partial \mathbf{w}} |_\mathbf{w = w_t}$ 
4. Environment provides new state $s_{t+1}$ and reward $r_t$.
5. Compute TD target: $y_t = r_t + \gamma \cdot \mathop{max}\limits_a Q(s_{t+1},a;\mathbf{W_t})$. 
6. Gradient descent: $\mathbf{w}_{t+1} = \mathbf{w}_t - \alpha \cdot (q_t - y_t) \cdot \mathbf{d}_t$
