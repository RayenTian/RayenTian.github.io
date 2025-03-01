---
title: Basic Concept
date: 2025-03-01 14:32:38
tags: [RL, LLM, ShusenWang]
mathjax: true
---

# Basic concept

## Preknowledge

1. Random Variable： a variable whose values depend on outcomes of a random event.

2. Probability Density Function(PDF):概率密度函数

   e.g.
   $$
   Gaussian distribution
   p(x) = \frac{1}{\sqrt{2\pi \sigma^2}}exp(-\frac{(x - \mu)^2}{2\sigma^2})
   $$

3. Expectation

4. Random Sampling

## Terminologies

1. state $s$
1. action $a$ 
1. policy $\pi$: is a PDF. $\pi (s,a) \mapsto [0,1]: \pi(a|s) = \mathbb{P}(A = a|S = s)$
1. reward $R$  
1. state transition: old state ->(action) new state: $p(s'| s,a) = \mathbb{P}(S' = s' | S = s, A = a)$. Only the environment know, the user can not know it.

### Randomness in Reinforcement Learning

1. Action have randomness($\pi$ sample)

2. State transition have randomness($p$ sample)

### Rewards and Return

1. Return (aka cumulative future reward)

   $U_t = R_t + R_{t+1} + R_{t+2} + R{t+3}.....$

2. Discount return (aka cumulative discounted future reward)

   - $\gamma$: discount rate
   - $U_t = R_t + \gamma R_{t+1} +\gamma^2 R_{t+2} + \gamma^3 R{t+3}.....$

   - At time step $t$, the return $U_t$ is ==random==.

     - Action can be random $\mathbb{P}(A = a | S = s) = \pi(a|s)$

     - New state can be random $\mathbb{P}(S' = s' | S = s, A = a) = p(s' | s,a).$
     - For any $i \geq t$, the reward $R_i$ depends on $S_i$ and $A_i$. Thus, given $s_t$, the return $U_t$ depends on the random variables: $A_t, A_{t+1}, A_{t+2}...$ and $S_{t+1}, S_{t+2}...$ 

3. Action-Value Function $Q(s,a)$ for policy $\pi$: how good it is for an agent to pick action a while being in state $s$

   $Q_\pi(s_t,a_t) = \mathbb{E}[U_t|S_t = s_t, A_t = a_t]$ 

4. Optimal action-value function

   $Q^*(s_t,a_t) = \mathop{max}\limits_{\pi} Q_\pi(s_t,a_t).$ 

5. State-value function: how good the situation is in state $s$

   $V\pi(s_t) = \mathbb{E}_A[Q_\pi(s_t,A)] = \Sigma_a \pi(a|s_t) \cdot Q_\pi(s_t,a).$

   $V\pi(s_t) = \mathbb{E}_A[Q_\pi(s_t,A)] = \int \pi (a|s_t)\cdot Q_\pi(s_t,a) \,da.$ 

6. $\mathbb{E}_S[V_\pi(S)]$ evaluates how good the pollicy $\pi$ is.

## How does AI control the agent?

Upon observe the state $s_t$:

1. Suppose we have a good policy $\pi(a|s)$.
   - random sampling: $a_t \sim \pi(\cdot | s_t)$
2. Suppose we know the optimal action-value function $Q^*(s,a)$.
   - choose the action that maximizes the value: $a_t = argmax_a Q^*(s_t,a).$



