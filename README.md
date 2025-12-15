# Reinforcement Learning Course Exercises

![Project Status](https://img.shields.io/badge/Status-Completed-success)
![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20RL-red)

## 📌 Overview

This repository contains my solutions and implementations for the exercises assigned during my Reinforcement Learning course. The projects progress from fundamental tabular methods to advanced deep reinforcement learning algorithms for continuous control.

Each exercise is organized into its own directory (`ex1`, `ex2`, etc.) containing the relevant Jupyter Notebooks.

---

## 📂 Repository Structure & Contents

### [ex1/](ex1/) - Introduction & Basics
* **`ex1.ipynb`**: Introduction to RL environments and basic probability concepts. Likely covers Multi-armed Bandits or basic Dynamic Programming foundations.

### [ex2/](ex2/) - Tabular Methods (Planning)
* **`ex2.ipynb`**: Implementation of classical Dynamic Programming algorithms, such as **Policy Iteration** and **Value Iteration**, to solve finite Markov Decision Processes (MDPs).

### [ex3/](ex3/) - Model-Free Prediction & Control
* **`ex3.ipynb`**: Introduction to Monte Carlo methods and Temporal Difference (TD) learning.
    * Implementations likely include **SARSA** and **Q-Learning** for discrete environments.

### [ex4/](ex4/) - Value Function Approximation
Moving beyond tabular methods to handle large state spaces.
* **`ex4_rbf.ipynb`**: Linear function approximation using **Radial Basis Functions (RBFs)** and coarse coding.
* **`ex4_dqn.ipynb`**: Implementation of **Deep Q-Networks (DQN)** with Experience Replay and Target Networks to solve environments with high-dimensional observations.

### [ex5/](ex5/) - Advanced Value Methods
* **`ex5.ipynb`**: Exploration of advanced topics such as n-step Bootstrapping, Eligibility Traces ($\lambda$-return), or planning methods like Dyna-Q.

### [ex6/](ex6/) - Policy Gradients & Continuous Control
Solving environments with continuous action spaces.
* **`ex6_PG_AC.ipynb`**: Implementation of **Policy Gradient** methods (e.g., REINFORCE) and **Actor-Critic** architectures.
* **`ex6_DDPG.ipynb`**: Implementation of **Deep Deterministic Policy Gradient (DDPG)** for continuous control tasks (e.g., MuJoCo or Box2D environments).

---

## 🛠️ Tech Stack & Requirements

The solutions are implemented in **Python** using **Jupyter Notebooks**.

**Key Libraries:**
* `numpy` (Matrix operations)
* `matplotlib` (Plotting learning curves)
* `gym` / `gymnasium` (RL Environments)
* `torch` (PyTorch) or `tensorflow` (Deep Learning models)

### Installation
To run these notebooks locally, ensure you have the required dependencies installed:

```bash
pip install numpy matplotlib gymnasium torch jupyter