"""Controles del contrato del entrenamiento experimental V2."""
from pathlib import Path
import sys
import unittest
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))
from entrenar import prepare_v2, feature_groups_v2

class V2TrainingTests(unittest.TestCase):
    def test_feature_groups_are_separate(self):
        groups = feature_groups_v2()
        self.assertEqual(len(groups['favorabilidad']), 5)
        self.assertEqual(len(groups['combinadas']), 15)
        self.assertEqual(len(set(groups['combinadas'])), 15)

    def test_official_population_and_partitions(self):
        data_dir = ROOT/'data/processed'
        path = data_dir/'v2/muestras_v2.csv'
        if not path.exists():
            self.skipTest('Ejecutar primero procesamiento V2')
        frames, partitions, groups = prepare_v2(path, data_dir/'particiones.json')
        self.assertEqual([len(frames[(0, 7)]), len(frames[(0, 14)])], [428, 428])
        self.assertEqual([len(frames[(7, 7)]), len(frames[(7, 14)])], [362, 362])

    def test_reject_modified_target(self):
        import tempfile
        from shutil import copyfile
        data_dir = ROOT/'data/processed'
        path = data_dir/'v2/muestras_v2.csv'
        if not path.exists():
            self.skipTest('Ejecutar primero procesamiento V2')
        with tempfile.TemporaryDirectory() as directory:
            tmp = Path(directory)
            copyfile(data_dir/'muestras.csv', tmp/'muestras.csv')
            copyfile(data_dir/'particiones.json', tmp/'particiones.json')
            df = pd.read_csv(path)
            df.loc[0, 'y'] = 1-int(df.loc[0, 'y'])
            df.to_csv(tmp/'alterado.csv', index=False)
            with self.assertRaisesRegex(ValueError, 'V2 modificó'):
                prepare_v2(tmp/'alterado.csv', tmp/'particiones.json')

if __name__ == '__main__':
    unittest.main()
