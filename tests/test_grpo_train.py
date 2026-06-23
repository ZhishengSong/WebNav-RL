from __future__ import annotations

import unittest

try:
    import torch
except ModuleNotFoundError:
    torch = None

if torch is not None:
    from training.grpo_train import IGNORE_INDEX, sequence_kl_penalty, sequence_policy_loss


@unittest.skipIf(torch is None, "torch is an optional model-training dependency")
class GrpoTrainTests(unittest.TestCase):
    def test_positive_advantage_rewards_higher_target_probability(self) -> None:
        labels = torch.tensor([[IGNORE_INDEX, 1, 2]])
        weak_logits = torch.zeros((1, 3, 4))
        strong_logits = torch.zeros((1, 3, 4))
        strong_logits[0, 0, 1] = 4.0
        strong_logits[0, 1, 2] = 4.0
        advantage = torch.tensor([1.0])

        weak_loss = sequence_policy_loss(weak_logits, labels, advantage, clip_advantage=1.0)
        strong_loss = sequence_policy_loss(strong_logits, labels, advantage, clip_advantage=1.0)

        self.assertLess(float(strong_loss), float(weak_loss))

    def test_kl_penalty_is_zero_when_logits_match(self) -> None:
        labels = torch.tensor([[IGNORE_INDEX, 1, 2]])
        logits = torch.zeros((1, 3, 4))

        kl_loss, policy_log_prob, reference_log_prob = sequence_kl_penalty(logits, logits, labels)

        self.assertAlmostEqual(float(kl_loss), 0.0, places=6)
        self.assertAlmostEqual(float(policy_log_prob), float(reference_log_prob), places=6)


if __name__ == "__main__":
    unittest.main()
