import numpy as np
import matplotlib.pyplot as plt


def transition_kernel(env):
    """
    Returns:
        P: transition kernel
        nS: number of states
        nA: number of actions
    """
    P = env.P
    nS = env.observation_space.n
    nA = env.action_space.n
    return P, nS, nA

# ------ Value iteration helper ------

def bellman_opt_backup(V: np.ndarray, P, nS: int, nA: int, gamma: float):
    """
    Single full sweep Bellman optimality backup.
    Returns:
        V_new: shape (nS,)
        Q: shape (nS, nA)  
    """
    Q = np.zeros((nS, nA), dtype=float)
    for s in range(nS):
        for a in range(nA):
            val = 0.0
            for (p, s_next, r, done) in P[s][a]:
                val += p * (r + gamma * (0.0 if done else V[s_next]))
            Q[s, a] = val
    V_new = Q.max(axis=1)
    return V_new, Q


def value_iteration(P, nS: int, nA: int, gamma: float, 
                    tol: float, max_iters: int = 100):
    """
    Returns:
        V: optimal values (nS,)
        policy: greedy policy w.r.t V (nS,)
        Q: optimal Q-values (nS, nA)
        diff: L2 norm of value function difference after update
    """
    V = np.zeros(nS, dtype=float)
    Q = np.zeros((nS, nA), dtype=float)
    diff = []

    for _ in range(max_iters):
        V_new, Q_new = bellman_opt_backup(V, P, nS, nA, gamma)

        delta = np.linalg.norm(V_new - V, ord=2)
        diff.append(delta)
        V, Q = V_new, Q_new
        if delta <= tol:
            break

    policy = np.argmax(Q, axis=1)
    return V, policy, Q, diff

# ------ Policy iteration helper ------

def policy_evaluation(P, nS: int, nA: int, policy: np.ndarray,
                      gamma: float, theta: float, max_iters: int = 100):
    """
    Returns:
        V: value function for policy
    """
    V = np.zeros(nS, dtype=float)

    for _ in range(max_iters):
        V_new = V.copy()
        for s in range(nS):
            a = int(policy[s])
            val = 0.0
            for (p, s_next, r, done) in P[s][a]:
                val += p * (r + gamma * (0.0 if done else V[s_next]))
            V_new[s] = val

        delta = np.linalg.norm(V_new - V, ord=2)
        V = V_new
        if delta < theta:
            break   
    return V


def action_value(P, nS: int, nA: int, V: np.ndarray, gamma: float):

    Q = np.zeros((nS, nA), dtype=float)
    for s in range(nS):
        for a in range(nA):
            val = 0.0
            for (p, s_next, r, done) in P[s][a]:
                val += p * (r + gamma * (0.0 if done else V[s_next]))
            Q[s, a] = val
    return Q


def greedy_policy(P, nS: int, nA: int, V: np.ndarray, gamma: float):
    """
    Greedy deterministic policy π(s) = argmax_a Q(s,a) w.r.t. V.
    """
    Q = action_value(P, nS, nA, V, gamma)
    pi = np.argmax(Q, axis=1)
    return pi, Q


def policy_iteration(P, nS: int, nA: int, gamma: float,
                     theta_eval: float, max_eval_iters: int = 10000,
                     max_pi_iters: int = 1000):
    """
    Classic Policy Iteration, terminate when policy is stable
    Returns:
        V_star: optimal state values
        pi_star: optimal deterministic policy
        Q_star: optimal action-values
        v_pi_deltas: list of L2 norm of value function updates across outer iterations
    """
    pi = np.zeros(nS, dtype=int)

    V_prev = np.zeros(nS, dtype=float)
    v_pi_deltas = []           

    for _ in range(max_pi_iters):
        # 1) Policy evaluation
        V_pi = policy_evaluation(P, nS, nA, pi, gamma, theta_eval, max_eval_iters)

        # record outer iteration delta
        v_pi_deltas.append(float(np.linalg.norm(V_pi - V_prev, ord=2)))
        V_prev = V_pi

        # 2) Greedy improvement
        pi_new, Q_pi = greedy_policy(P, nS, nA, V_pi, gamma)

        # 3) Check stability
        if np.array_equal(pi_new, pi):
            return V_pi, pi_new, Q_pi, v_pi_deltas

        pi = pi_new

    Q_pi = action_value(P, nS, nA, V_prev, gamma)
    return V_prev, pi, Q_pi, v_pi_deltas

# ------ Plotting helpers ------

def plot_convergence(deltas):

    plt.figure(figsize=(6,4))
    plt.plot(deltas, marker='o', linewidth=1)
    plt.yscale('log')
    plt.xlabel('Iteration k')
    plt.ylabel(r'$||V_k - V_{k-1}||_2$')
    plt.title('Value Iteration Convergence')
    plt.show()

def plot_pi_convergence(outer_deltas):
    """
    Plot ||V_{π_k} - V_{π_{k-1}}||_2 over outer PI iterations.
    """
    plt.figure(figsize=(6,4))
    plt.plot(outer_deltas, marker='o', linewidth=1)
    plt.yscale('log')
    plt.xlabel('Policy iteration k')
    plt.ylabel(r'$||V_{\pi_k}-V_{\pi_{k-1}}||_2$')
    plt.title('Policy Iteration Convergence (outer loop)')
    plt.show()

def plot_value_grid(V: np.ndarray, desc: np.ndarray, title='Optimal State Values'):

    nrow, ncol = desc.shape
    grid = V.reshape(nrow, ncol)
    plt.figure(figsize=(6,4))
    im = plt.imshow(grid, origin='upper')
    plt.colorbar(im, fraction=0.05, pad=0.05)
    for i in range(nrow):
        for j in range(ncol):
            plt.text(j, i, desc[i, j].decode('utf-8'), ha='center', va='center',
                     color='white', fontsize=10, fontweight='bold')
    plt.title(title)
    plt.xticks([])
    plt.yticks([])
    plt.show()

def plot_policy_arrows(policy: np.ndarray, desc: np.ndarray, title='Optimal Policy'):
    """
    Arrows indicate actions: 0:Left, 1:Down, 2:Right, 3:Up.
    """
    nrow, ncol = desc.shape
    arrow_chars = np.array(['←', '↓', '→', '↑'])
    grid = np.empty_like(desc, dtype='<U2')

    for s in range(policy.size):
        i, j = divmod(s, ncol)
        cell = desc[i, j].decode('utf-8')
        grid[i, j] = cell if cell in ['H', 'G'] else arrow_chars[policy[s]]

    plt.figure(figsize=(6, 4))
    plt.imshow(np.ones_like(desc, dtype=float), vmin=0, vmax=1)
    for i in range(nrow):
        for j in range(ncol):
            plt.text(j, i, grid[i, j],
                     ha='center', va='center', fontsize=10, fontweight='bold', color='black')
    plt.title(title)
    plt.xticks([]); plt.yticks([])
    plt.tight_layout()
    plt.show()
