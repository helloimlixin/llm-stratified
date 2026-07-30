import sys
from pathlib import Path
import unittest

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from pretrained_vqvae_imagegpt_pipeline import VQImageGPT, parse_patch_nums, split_multiscale_tokens  # noqa: E402


class TestPretrainedVqvaeImageGptPipeline(unittest.TestCase):
    def test_parse_patch_nums_supports_final_and_full_modes(self):
        self.assertEqual(parse_patch_nums("16"), (1, 16))
        self.assertEqual(parse_patch_nums("1,16"), (1, 16))
        self.assertEqual(parse_patch_nums("full"), (1, 2, 3, 4, 5, 6, 8, 10, 13, 16))

    def test_split_multiscale_tokens(self):
        tokens = torch.arange(2 * 20).view(2, 20)

        chunks = split_multiscale_tokens(tokens, (2, 4))

        self.assertEqual([tuple(chunk.shape) for chunk in chunks], [(2, 4), (2, 16)])
        self.assertTrue(torch.equal(chunks[0], tokens[:, :4]))
        self.assertTrue(torch.equal(chunks[1], tokens[:, 4:]))

    def test_inputs_from_targets_prepends_bos_and_shifts_right(self):
        model = VQImageGPT(vocab_size=11, seq_len=4, n_embd=8, n_head=2, n_layer=1)
        targets = torch.tensor([[3, 4, 5, 6]])

        inputs = model.inputs_from_targets(targets)

        self.assertEqual(inputs.tolist(), [[11, 3, 4, 5]])


if __name__ == "__main__":
    unittest.main()
