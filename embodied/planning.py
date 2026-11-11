import numpy as np
from .env import DockingEnv, pd_action


class LinearWorldModel:
    """Fit delta physical state from state/action; used for learned-model MPC."""
    def fit(self, dataset):
        current = dataset["observations"][:, :4]
        next_state = dataset["next_observations"][:, :4]
        features = np.c_[current, dataset["actions"], np.ones(len(current))]
        target = next_state - current
        episode_ids = dataset["episode_id"].astype(int)
        # Whole episodes held out; no neighboring transitions split across sets.
        ids = np.unique(episode_ids)
        validation_ids = np.random.default_rng(177).permutation(ids)[:max(1, len(ids) // 5)]
        valid = np.isin(episode_ids, validation_ids)
        train = ~valid
        self.weights = np.linalg.solve(features[train].T @ features[train] + 1e-4 * np.eye(7),
                                       features[train].T @ target[train])
        error = current[valid] + features[valid] @ self.weights - next_state[valid]
        return dict(heldout_state_rmse=float(np.sqrt(np.mean(error ** 2))),
                    train_transitions=int(train.sum()), validation_transitions=int(valid.sum()))

    def predict(self, states, actions):
        states = np.atleast_2d(states)
        features = np.c_[states, actions, np.ones(len(states))]
        out = states + features @ self.weights
        out[:, :2] = out[:, :2].clip(-1, 1)
        out[:, 2:] = out[:, 2:].clip(-1.5, 1.5)
        return out


class MPC:
    """CEM over constant-force candidates; replan after every real step."""
    def __init__(self, model, seed=7, horizon=15, candidates=128):
        self.model, self.horizon, self.candidates = model, horizon, candidates
        self.rng = np.random.default_rng(seed)

    def __call__(self, observation):
        mean, std = np.zeros(2), np.ones(2)
        goal = observation[4:6]
        for _ in range(2):
            actions = np.clip(self.rng.normal(mean, std, size=(self.candidates, 2)), -1, 1)
            actions[0] = 0
            states = np.repeat(observation[None, :4], self.candidates, axis=0)
            cost = np.zeros(self.candidates)
            for t in range(self.horizon):
                states = self.model.predict(states, actions)
                cost += .98 ** t * (np.square(states[:, :2] - goal).sum(-1)
                                    + .05 * np.square(states[:, 2:]).sum(-1)
                                    + .01 * np.square(actions).sum(-1))
            cost += 3 * np.square(states[:, :2] - goal).sum(-1)
            elite = actions[np.argsort(cost)[:16]]
            mean, std = elite.mean(0), elite.std(0) + .05
        return mean.astype(np.float32)
