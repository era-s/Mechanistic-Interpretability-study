"""Numerical regression tests; extract notebook helpers without loading GPT-2."""
import ast
import json
from pathlib import Path
import unittest
import torch

namespace = {'torch': torch}
notebook = json.loads((Path(__file__).resolve().parents[1] / 'notebooks/01_Circuits&ResidualStream.ipynb').read_text())
for cell in notebook['cells']:
    if cell['cell_type'] == 'code':
        source = ''.join(cell['source'])
        tree = ast.parse('\n'.join(line for line in source.splitlines() if not line.startswith('%')))
        for node in tree.body:
            if isinstance(node, ast.FunctionDef):
                exec(compile(ast.Module(body=[node], type_ignores=[]), '<notebook>', 'exec'), namespace)

class CircuitMathTests(unittest.TestCase):
    def helper(self, name):
        self.assertIn(name, namespace, f'Missing numerical helper: {name}')
        return namespace[name]

    @unittest.skipUnless(torch.backends.mps.is_available(), "MPS unavailable")
    def test_mps_weight_transfer_preserves_values(self):
        convert = self.helper('cpu_float64')
        result = convert(torch.tensor([1., 2., 3.], device='mps'))
        torch.testing.assert_close(result, torch.tensor([1., 2., 3.], dtype=torch.float64))

    def test_residual_diff_uses_initial_embedding(self):
        namespace['layers'] = range(2)
        cache = {
            'blocks.0.hook_resid_pre': torch.tensor([[[1.,2.]]]),
            'blocks.0.hook_resid_post': torch.tensor([[[2.,5.]]]),
            'blocks.1.hook_resid_pre': torch.tensor([[[2.,5.]]]),
            'blocks.1.hook_resid_post': torch.tensor([[[7.,11.]]]),
        }
        actual = self.helper('residual_delta')(cache)
        torch.testing.assert_close(actual, torch.tensor([[[1.,3.]], [[6.,9.]]]))

    def test_decode_snapshot_masks_history_and_updates_only_last_token(self):
        snapshot = self.helper('decode_snapshot')
        result = {
            'prefill_delta': torch.tensor([[[1.,2.]], [[3.,4.]]]),
            'decode_delta': torch.tensor([[[5.,6.],[7.,8.]], [[9.,10.],[11.,12.]]]),
        }
        early = snapshot(result, step=1, layer=0)
        late = snapshot(result, step=1, layer=1)
        torch.testing.assert_close(early, torch.tensor([[0.,0.],[0.,0.],[7.,8.]]))
        torch.testing.assert_close(late, torch.tensor([[0.,0.],[0.,0.],[11.,12.]]))
        torch.testing.assert_close(early[:-1], late[:-1])
        baseline = snapshot(result, step=0, layer=-1)
        torch.testing.assert_close(baseline, torch.tensor([[0.,0.],[0.,0.]]))

    def test_squared_spectrum_metrics(self):
        metrics = self.helper('rank_metrics')
        # sigma^2 = [3, 1], hence p=[0.75, 0.25]; catches sigma vs sigma^2.
        s = torch.tensor([3.0, 1.0], dtype=torch.float64).sqrt()
        result = metrics(s)
        self.assertAlmostEqual(result['Entropy rank'].item(), 1.7547653506033232)
        self.assertAlmostEqual(result['Participation ratio'].item(), 1.6)
        self.assertEqual(result['95% dimension'].item(), 2)
        for key, value in result.items():
            torch.testing.assert_close(metrics(s * 1e-100)[key], value)
            torch.testing.assert_close(metrics(s * 1e100)[key], value)

    def test_rank_metrics_zero_uniform_and_variance_threshold(self):
        metrics = self.helper('rank_metrics')
        s = torch.tensor([[1.,1.,1.,1.], [4.,0.,0.,0.], [0.,0.,0.,0.]], dtype=torch.float64)
        result = metrics(s)
        for key in ['Entropy rank', 'Participation ratio']:
            torch.testing.assert_close(result[key], torch.tensor([4.,1.,0.], dtype=torch.float64))
        self.assertEqual(result['95% dimension'].tolist(), [4,1,0])
        for energy, want in [([95.,5.],1), ([94.,6.],2), ([1.,99.],1)]:
            self.assertEqual(metrics(torch.tensor(energy, dtype=torch.float64).sqrt())['95% dimension'].item(), want)

    def test_full_projection_preserves_head_concatenation(self):
        assemble = self.helper('full_projection_matrices')
        # Two heads of width two, residual width four, in a single layer.
        q = torch.arange(16, dtype=torch.float64).reshape(1,2,2,4)
        o = torch.arange(16, dtype=torch.float64).reshape(1,2,4,2)
        full = assemble({'Q':q, 'K':q, 'V':q, 'O':o})
        expected_q = torch.arange(16,dtype=torch.float64).reshape(4,4)
        expected_o = torch.tensor([[0,1,8,9],[2,3,10,11],[4,5,12,13],[6,7,14,15]], dtype=torch.float64)
        for key in ['Q','K','V']:
            torch.testing.assert_close(full[key][0], expected_q)
        torch.testing.assert_close(full['O'][0], expected_o)
        self.assertTrue(all(w.shape == (1,4,4) for w in full.values()))

    def test_factored_singular_values_match_dense(self):
        svd = self.helper('product_singular_values')
        g = torch.Generator().manual_seed(7)
        a = torch.randn(2, 9, 3, generator=g, dtype=torch.float64)
        b = torch.randn(2, 3, 9, generator=g, dtype=torch.float64)
        expected = torch.linalg.svdvals(a @ b)[..., :3]
        torch.testing.assert_close(svd(a, b), expected)

    def test_eigenvectors_satisfy_full_operator(self):
        eig = self.helper('product_eigensystem')
        a = torch.tensor([[1.,0.],[0.,1.],[0.,0.]], dtype=torch.float64)
        b = torch.tensor([[0.,-1.,0.],[1.,0.,0.]], dtype=torch.float64)
        values, vectors = eig(a, b)
        self.assertTrue(torch.allclose(values.abs(), torch.ones(2, dtype=torch.float64)))
        torch.testing.assert_close((a @ b).to(torch.complex128) @ vectors, vectors * values)
        torch.testing.assert_close(vectors.norm(dim=0), torch.ones(2,dtype=torch.float64))

if __name__ == '__main__':
    unittest.main()
