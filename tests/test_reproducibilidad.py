import importlib.metadata as metadata
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from procesar_datos import ROOT, configured
from entrenar import run
from exportar_portal import export


class ReproducibilityTests(unittest.TestCase):
    def test_locked_packages_match_verified_environment(self):
        for line in (ROOT/'requirements-lock.txt').read_text().splitlines():
            if '==' in line:
                package,version=line.split('==')
                self.assertEqual(metadata.version(package),version,package)

    def test_training_refuses_to_overwrite_official_results(self):
        with self.assertRaises(ValueError):
            run(configured('datos'),configured('resultados'),ROOT/'config/proyecto.json')

    def test_export_contains_only_aggregates_and_figures(self):
        with tempfile.TemporaryDirectory() as temp:
            destination=Path(temp)/'public'
            export(destination)
            names=[str(p.relative_to(destination)) for p in destination.rglob('*')]
            for forbidden in ['specs','predicciones','joblib','ipynb','raw']:
                self.assertFalse(any(forbidden in name for name in names), forbidden)
            self.assertTrue((destination/'comparacion.csv').exists())


if __name__=='__main__':
    unittest.main()
