"""A probabilistic state-space model bridge, NOT an implementation of Dreamer."""
import json
import numpy as np
import torch
from torch import nn
from .env import Warehouse
from .tabular import value_iteration
from .replay import EpisodeStore
from .neural import setup


class StateWorldModel(nn.Module):
    """Predicts next-state distribution, reward and true continuation.

    One-hot state is already fully observed. A later pixel/RSSM model must replace
    this representation, not pretend this classifier is a Dreamer latent model.
    """
    def __init__(self):
        super().__init__()
        self.body = nn.Sequential(nn.Linear(74, 96), nn.ELU(), nn.Linear(96, 96), nn.ELU())
        self.next_state = nn.Linear(96, 70)
        self.reward = nn.Linear(96, 1)
        self.continue_logit = nn.Linear(96, 1)

    def forward(self, states, actions):
        s = nn.functional.one_hot(states.long(), 70).float()
        a = nn.functional.one_hot(actions.long(), 4).float()
        features = self.body(torch.cat([s, a], dim=-1))
        return (self.next_state(features), self.reward(features).squeeze(-1),
                self.continue_logit(features).squeeze(-1))

    def imagine(self, initial_states, action_sequence):
        """Sample model trajectories [T,B]; no true environment calls."""
        states, rewards, continues = [initial_states], [], []
        for actions in action_sequence:
            logits, reward, cont = self(states[-1], actions)
            states.append(torch.distributions.Categorical(logits=logits).sample())
            rewards.append(reward)
            continues.append(cont.sigmoid())
        return dict(states=torch.stack(states), rewards=torch.stack(rewards),
                    continuation=torch.stack(continues))


def tensors(transitions, device="cpu"):
    return (torch.tensor([x.state for x in transitions], device=device),
            torch.tensor([x.action for x in transitions], device=device),
            torch.tensor([x.next_state for x in transitions], device=device),
            torch.tensor([x.reward for x in transitions], device=device),
            torch.tensor([not x.terminated for x in transitions], dtype=torch.float32, device=device))


def experiment(episodes, updates, seed, shaping, output, device="auto"):
    device = setup(seed, device)
    rng = np.random.default_rng(seed)
    env = Warehouse(seed=seed, shaping=shaping)
    _, reference_q = value_iteration(env)
    store = EpisodeStore(seed=seed)
    for ep in range(max(20, episodes)):
        state, trajectory = env.reset(), []
        # Half pure random, half exploring reference: dataset collection only.
        epsilon = 1.0 if ep % 2 else 0.3
        while True:
            action = int(rng.integers(4)) if rng.random() < epsilon else int(reference_q[state].argmax())
            tr = env.step(action)
            trajectory.append(tr); state = tr.next_state
            if tr.terminated or tr.truncated:
                break
        store.append(trajectory)
    store.save(output / "episodes.json")
    # Split whole episodes. Neighboring transitions never leak across the split.
    order = rng.permutation(len(store.episodes))
    cut = int(len(order) * 0.8)
    train = [tr for i in order[:cut] for tr in store.episodes[i]]
    valid = [tr for i in order[cut:] for tr in store.episodes[i]]
    model = StateWorldModel().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-3)
    for _ in range(updates):
        batch = [train[i] for i in rng.integers(len(train), size=128)]
        s, a, ns, reward, cont = tensors(batch, device)
        logits, predicted_reward, predicted_cont = model(s, a)
        loss = (nn.functional.cross_entropy(logits, ns)
                + 0.1 * nn.functional.mse_loss(predicted_reward, reward)
                + nn.functional.binary_cross_entropy_with_logits(predicted_cont, cont))
        optimizer.zero_grad(); loss.backward(); optimizer.step()
    with torch.no_grad():
        s, a, ns, reward, cont = tensors(valid, device)
        logits, predicted_reward, predicted_cont = model(s, a)
        metrics = dict(heldout_next_state_nll=float(nn.functional.cross_entropy(logits, ns)),
                       heldout_reward_rmse=float((predicted_reward - reward).square().mean().sqrt()),
                       heldout_continuation_accuracy=float(((predicted_cont > 0) == cont.bool()).float().mean()),
                       heldout_top1_state_accuracy=float((logits.argmax(-1) == ns).float().mean()),
                       train_transitions=len(train), validation_transitions=len(valid),
                       updates=updates, episodes=len(store.episodes),
                       scope="one-step probabilistic model; no latent RSSM or learned imagination actor")
        imagined = model.imagine(torch.tensor([env.reset()], device=device),
                                 torch.tensor([[2], [2], [2], [2], [1], [1]], device=device))
        example = {k: v.tolist() for k, v in imagined.items()}
    (output / "imagination_example.json").write_text(json.dumps(example, indent=2), encoding="utf-8")
    # Exercise the sequence API used by a future RSSM.
    sequences = store.sample_sequences(8, 16)
    np.savez_compressed(output / "sequence_sample.npz", **sequences)
    torch.save(model.state_dict(), output / "world_model.pt")
    return metrics
