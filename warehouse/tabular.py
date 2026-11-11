import numpy as np
from .env import Warehouse, evaluate


def bellman_q(env, value, state, action):
    return sum(p * (reward + env.gamma * (not terminal) * value[ns])
               for p, ns, reward, base, terminal in env.outcomes(state, action))


def value_iteration(env, tolerance=1e-10):
    value = np.zeros(env.n_states)
    q = np.zeros((env.n_states, env.n_actions))
    for _ in range(3000):
        for s in env.valid_states:
            q[s] = [bellman_q(env, value, s, a) for a in range(4)]
        new_value = q.max(axis=1)
        error = np.max(np.abs(new_value - value))
        value = new_value
        if error < tolerance:
            return value, q
    raise RuntimeError("Bellman iteration failed to converge.")


def policy_evaluation(env, probabilities):
    # Exact Bellman expectation solution (I - gamma P_pi)V = r_pi.
    matrix, rewards = np.eye(env.n_states), np.zeros(env.n_states)
    for s in env.valid_states:
        for a in range(4):
            for p, ns, reward, base, terminal in env.outcomes(s, a):
                weight = probabilities[s, a] * p
                rewards[s] += weight * reward
                if not terminal:
                    matrix[s, ns] -= env.gamma * weight
    return np.linalg.solve(matrix, rewards)


def choose(q, state, epsilon, rng):
    if rng.random() < epsilon:
        return int(rng.integers(4))
    row = q[state]
    return int(rng.choice(np.flatnonzero(row == row.max())))


def control(name, episodes, seed, shaping=True):
    env = Warehouse(seed=seed, shaping=shaping)
    rng = np.random.default_rng(seed)
    q = np.zeros((env.n_states, 4))
    logs, steps = [], 0
    for episode in range(episodes):
        epsilon = max(0.05, 0.9 * 0.997 ** episode)
        state = env.reset()
        action = choose(q, state, epsilon, rng)
        for _ in range(env.max_steps):
            tr = env.step(action)
            # SARSA: next sampled behavior action. Q-learning: max over next actions.
            next_action = choose(q, tr.next_state, epsilon, rng)
            next_value = (q[tr.next_state, next_action] if name == "sarsa"
                          else q[tr.next_state].max())
            target = tr.reward + env.gamma * (not tr.terminated) * next_value
            delta = target - q[state, action]
            q[state, action] += 0.12 * delta
            steps += 1
            state, action = tr.next_state, next_action
            if tr.terminated or tr.truncated:
                break
        if (episode + 1) % max(1, episodes // 20) == 0:
            score = evaluate(lambda s: int(q[s].argmax()), episodes=50, shaping=shaping)
            logs.append(dict(episode=episode + 1, steps=steps, **score))
    return q, logs


def prediction(name, episodes, seed, shaping=True):
    """MC and TD evaluate the SAME fixed policy; neither improves it here."""
    # Complete episodes for unbiased MC returns; a high guard catches a broken
    # policy instead of silently treating a timeout as a zero-value terminal.
    env = Warehouse(seed=seed, shaping=shaping, max_steps=10000)
    rng = np.random.default_rng(seed)
    _, oracle_q = value_iteration(env)
    probabilities = np.full((env.n_states, 4), 0.1 / 4)
    probabilities[np.arange(env.n_states), oracle_q.argmax(axis=1)] += 0.9
    exact = policy_evaluation(env, probabilities)
    value = np.zeros(env.n_states)
    counts = np.zeros(env.n_states)
    logs, visits, steps = [], np.zeros(env.n_states), 0
    for episode in range(episodes):
        state, trajectory = env.reset(), []
        for _ in range(env.max_steps):
            action = int(rng.choice(4, p=probabilities[state]))
            tr = env.step(action)
            trajectory.append(tr)
            visits[state] += 1
            if name == "td":
                delta = tr.reward + env.gamma * (not tr.terminated) * value[tr.next_state] - value[state]
                value[state] += 0.05 * delta
            state = tr.next_state
            steps += 1
            if tr.terminated or tr.truncated:
                break
        if trajectory[-1].truncated:
            raise RuntimeError("Prediction episode did not terminate; increase guard.")
        if name == "mc":
            # Complete episodes only: no fabricated zero terminal return at a timeout.
            if trajectory[-1].terminated:
                returns, g = [], 0.0
                for tr in reversed(trajectory):
                    g = tr.reward + env.gamma * g
                    returns.append(g)
                seen = set()
                for tr, g in zip(trajectory, reversed(returns)):
                    if tr.state not in seen:  # first-visit MC sample average
                        seen.add(tr.state)
                        counts[tr.state] += 1
                        value[tr.state] += (g - value[tr.state]) / counts[tr.state]
        if (episode + 1) % max(1, episodes // 20) == 0:
            mask = visits >= 20
            logs.append(dict(episode=episode + 1, steps=steps,
                             visited_state_rmse=float(np.sqrt(np.mean((value[mask] - exact[mask]) ** 2))),
                             start_estimate=float(value[env.encode(0, 0, 0)]),
                             start_exact=float(exact[env.encode(0, 0, 0)])))
    return value, logs, probabilities
