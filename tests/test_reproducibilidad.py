import importlib.metadata as metadata
from pathlib import Path
import sys
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from procesar_datos import ROOT, configured
from entrenar import run



class ReproducibilityTests(unittest.TestCase):
    def test_locked_packages_match_verified_environment(self):
        for line in (ROOT/'requirements.txt').read_text().splitlines():
            if '==' in line:
                package,version=line.split('==')
                self.assertEqual(metadata.version(package),version,package)

    def test_training_refuses_to_overwrite_official_results(self):
        with self.assertRaises(ValueError):
            run(configured('datos'),configured('resultados'),ROOT/'scripts/proyecto.json')

    def test_public_downloads_match_sources(self):
        public = ROOT/'web/public/academico'
        for source in [ROOT/'reports/INFORME.md', *(ROOT/'notebooks').glob('*.ipynb')]:
            self.assertEqual(source.read_bytes(), (public/source.name).read_bytes())
        for item in (ROOT/'web/public').rglob('*'):
            self.assertNotIn(item.suffix, {'.joblib', '.csv'})
            self.assertFalse({'models', 'specs', 'predictions', 'raw'}.intersection(item.parts))
