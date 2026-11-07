import csv
import tempfile
import unittest
from pathlib import Path

from scripts.seed_summary import collect_runs, permutation_p, summarize


def write(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


class SeedSummaryTests(unittest.TestCase):
    def test_permutation_counts_both_tails(self):
        # 6 repartos de 2+2; solo {1,2}|{3,4} y {3,4}|{1,2} alcanzan la diferencia observada.
        self.assertAlmostEqual(permutation_p([1, 2], [3, 4]), 2 / 6)
        self.assertEqual(permutation_p([1, 1], [1, 1]), 1.0)

    def test_runs_join_both_sources_and_summary_uses_all_seeds(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for folder, seeds in (('grupo', [0]), ('replicas', [0, 1])):
                yolo, per_class, overall = [], [], []
                for name in ('B1', 'M2'):
                    for seed in seeds:
                        value = 0.6 + 0.01 * seed + (0.02 if name == 'M2' else 0)
                        for cls in ('all', 'gun'):
                            yolo.append({'experiment': name, 'seed': seed, 'class': cls,
                                         'map50': value + 0.2, 'map50_95': value})
                        per_class.append({'experiment': name, 'seed': seed, 'class': 'gun', 'fn': 100 + seed})
                        overall.append({'experiment': name, 'seed': seed, 'small_gun_recall': 0.1})
                write(root / folder / 'validacion_yolov5.csv', yolo)
                write(root / folder / 'per_class.csv', per_class)
                write(root / folder / 'comparison.csv', overall)
            runs = collect_runs(root=root)
            self.assertEqual(len(runs), 6)
            self.assertEqual({r['device'] for r in runs}, {'A100', 'RTX 4090'})
            summary = {s['metric']: s for s in summarize(runs)}
            self.assertEqual(summary['map50_95']['n_B1'], 3)
            self.assertAlmostEqual(summary['map50_95']['difference'], 0.02)


if __name__ == '__main__':
    unittest.main()
