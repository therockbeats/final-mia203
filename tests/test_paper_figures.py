import csv
import tempfile
import unittest
from pathlib import Path

from scripts.paper_figures import crop, missed_small_guns


class PaperFigureTests(unittest.TestCase):
    def test_crop_stays_inside_image_and_contains_box(self):
        left, top, right, bottom = crop(1280, 720, [1263, 90, 1280, 106])
        self.assertEqual((right, bottom), (1280, 188))
        self.assertTrue(left <= 1263 and top <= 90)

    def test_selection_keeps_one_small_unmatched_gun_per_video(self):
        rows = [
            {'video': 'a', 'area_relative_gt': 0.002, 'category': 'sin_candidato'},
            {'video': 'a', 'area_relative_gt': 0.001, 'category': 'sin_candidato'},
            {'video': 'b', 'area_relative_gt': 0.003, 'category': 'localizacion'},
            {'video': 'c', 'area_relative_gt': 0.020, 'category': 'sin_candidato'},
            {'video': 'd', 'area_relative_gt': 0.004, 'category': 'sin_candidato'},
        ]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'diag.csv'
            with path.open('w', encoding='utf-8', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=['experiment', 'type', 'image', 'video', 'category', 'area_relative_gt'])
                writer.writeheader()
                for i, r in enumerate(rows):
                    writer.writerow({'experiment': 'M2', 'type': 'FN', 'image': f'{i}.jpg', **r})
            chosen = missed_small_guns(path)
        self.assertEqual([(r['video'], r['area_relative_gt']) for r in chosen], [('a', '0.001'), ('d', '0.004')])


if __name__ == '__main__':
    unittest.main()
