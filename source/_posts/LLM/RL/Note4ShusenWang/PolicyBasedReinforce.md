---
title: Policy Based Reinforce
date: 2025-03-01 14:34:25
tags: [RL, LLM, ShusenWang]
mathjax: true
---

# Policy Based Reinforce

## Policy Network $\pi(a|s;\theta)$

Policy Network: Use a neural net to approximate $\pi(a|s)$. 

- Use policy network $\pi(a|s;\mathbf{\theta})$ to approximate $\pi(a|s)$.
- $\theta$: trainable parameters of the neural net.
- $\Sigma_{a\in A} \pi(a|s;\theta) = 1$ 

## Policy-Based Reinforcement Learning

==State-value function==: how good the situation is in state $s$

- $V\pi(s_t) = \mathbb{E}_A[Q_\pi(s_t,A)] = \Sigma_a \pi(a|s_t) \cdot Q_\pi(s_t,a).$

- $V\pi(s_t) = \mathbb{E}_A[Q_\pi(s_t,A)] = \int \pi (a|s_t)\cdot Q_\pi(s_t,a) \,da.$ 

Approximate state-value function:

- Approximate policy function $\pi(a|s_t)$ by policy network $\pi(a|s_t;\theta)$.
- Appproximate value function $V_\pi(s_t)$ by: $V\pi(s_t) = \Sigma_a \pi(a|s_t;\theta) \cdot Q_\pi(s_t,a).$ 

$$
V(s;\theta) = \Sigma_a \pi(a|s;\theta) \cdot Q_\pi(s,a).
$$

==Policy-Based learning==: Learn $\theta$ that maximizes $J(\theta) = \mathbb{E}_S[V(S;\theta)]$. 

## Policy gradient ascent

- Observe state s.
- Update policy by: $\theta \leftarrow \theta + \beta \cdot \frac{\partial V(s;\theta)}{\partial \theta}$

### Policy Gradient

$$
\begin{align}
\frac{\partial V(s;\theta)}{\partial \theta} &= \frac{ \Sigma_a \partial \pi(a|s;\theta) \cdot Q_\pi(s,a) }{\partial \theta} \\
&=\Sigma_a  \frac{ \partial \pi(a|s;\theta) }{\partial \theta}\cdot Q_\pi(s,a) \quad (Pretend \  Q_\pi \  is \  independent \  of \  \theta)\\
&= \Sigma_a \pi(a|s;\theta) \cdot \frac{ \partial log \pi(a|s;\theta) }{\partial \theta} \cdot Q_\pi(s,a) \\
&= \mathbb{E}_A[\frac{ \partial log \pi(a|s;\theta) }{\partial \theta} \cdot Q_\pi(s,A) ]
\end{align}
$$

### Calculate Policy Gradient for Discrete Actions

1. If the actions are ==discrete==, e.g. action space $\mathcal{A}$ = \{"left","right","up"\},...

   Use $\frac{\partial V(s;\theta)}{\partial \theta} = \Sigma_a  \frac{ \partial \pi(a|s;\theta) }{\partial \theta}\cdot Q_\pi(s,a)$ 

   1. Calculate $f(a,\theta) =  \frac{ \partial \pi(a|s;\theta) }{\partial \theta}\cdot Q_\pi(s,a) $ for every action $a \in \mathcal{A}$. 
   2. Policy gradient: $\frac{\partial V(s;\theta)}{\partial \theta} = f(left,\theta) + f(right,\theta) + f(up,\theta)$

2. If the actions are ==continuous==, e.g. action space $\mathcal{A}$ = [0,1]. 

   Use $\frac{\partial V(s;\theta)}{\partial \theta} = \mathbb{E}_{A\sim\pi(\cdot|s;\theta)}[\frac{ \partial log \pi(a|s;\theta) }{\partial \theta} \cdot Q_\pi(s,A) ]$  ==It's difficult to calculate the $\int$ of $\pi$. So we use Monte Carlo Approximation.==

   1. Randomly sample an action $\hat{a}$ according to the PDF $\pi(\cdot | s;\theta)$ 
   2. Calculate $g(\hat{a},\theta) = \frac{ \partial log \pi(\hat{a}|s;\theta) }{\partial \theta} \cdot Q_\pi(s,\hat{a})$. 

   > [!NOTE]
   >
   > Obviously, $\mathbb{E}_A[g(A,\theta)] = \frac{\partial V(s;\theta)}{\partial \theta}$.
   >
   > $g(\hat{a},\theta)$ is an unbiased estimate of $\frac{\partial V(s;\theta)}{\partial \theta}$.

   3. Use $g(\hat{a},\theta)$ as an approximation to the policy gradient $\frac{\partial V(s;\theta)}{\partial \theta}$. (This is called Monte Carlo approximation. The unbiased estimate is important.) 

   > [!IMPORTANT]
   >
   > This approach also works for ==discrete== actions.


##  Algorithm

1. Observe the state $s_t$.
2. Randomly sample action $a_t$ according to $\pi(\cdot | s_t;\theta_t)$ 
3. Compute $q_t \approx Q_\pi(s_t,a_t)$ (some estimate).
4. Differentiate policy network: $\mathbf{d}_{\theta,t} = \frac{\partial log \ \pi(a_t | s_t, \theta)}{\partial \theta} |_{\theta = \theta_t}$
5. (Approximate) policy gradient: $\mathbf{g}(\hat{a},\theta) = q_t \cdot \mathbf{d_{\theta,t}}$ .
6. Update policy network: $\theta_{t+1} = \theta_t + \beta \cdot \mathbf{g}(a_t, \theta_t)$.

### Compuet  $q_t \approx Q_\pi(s_t,a_t)$ 

1. Option 1: REINFORCE

   1. Play the game to the end and generate the trajectory: $s_1,a_1,r_1,s_2,a_2,r_2,...,s_T,a_T,r_T$.
   2. Compute the discounted return $u_t = \Sigma_{k = t}^T \gamma^{k-t}r_k$, for all t.
   3. Since $Q_\pi(s_t,a_t) = \mathbb{E}[U_t]$, we can use $u_t$ to approximate $Q_\pi(s_t,a_t)$.
   4. $\rightarrow$ Use $q_t = u_t$

2. Option 2: Approximate $Q_\pi$ using a neural network.

   This lead to the actor-critic method.

   
