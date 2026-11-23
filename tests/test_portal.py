"""El portal conserva los valores y las poblaciones del experimento oficial."""
import json
from pathlib import Path
import unittest
import zipfile
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]


class PortalTests(unittest.TestCase):
    def test_report_download_contains_only_the_existing_report(self):
        public=ROOT/'web/public/academico'
        with zipfile.ZipFile(public/'informe-academico.zip') as archive:
            self.assertEqual(archive.namelist(),['INFORME.html'])
            self.assertEqual(archive.read('INFORME.html'),(public/'INFORME.html').read_bytes())

    def test_official_metrics_and_selection(self):
        data=json.loads((ROOT/'web/app/portal-data.json').read_text(encoding='utf-8'))
        official=pd.read_csv(ROOT/'results/comparacion.csv').set_index('config')
        frozen=json.loads((ROOT/'results/seleccion.json').read_text(encoding='utf-8'))['selected']
        self.assertEqual({m['id'] for m in data['models']},set(official.index))
        for model in data['models']:
            row=official.loc[model['id']]
            for key,column in [('cv','cv_f1'),('f1','historico_f1'),('precision','historico_precision'),('recall','historico_recall')]:
                self.assertEqual(model[key],row[column])
            self.assertEqual(model['selected'],frozen[str(model['horizon'])]==model['id'])
            self.assertEqual(sum(sum(r) for r in model['matrix']),model['n'])
            self.assertEqual(model['matrix'],[[int(row.historico_tn),int(row.historico_fp)],[int(row.historico_fn),int(row.historico_tp)]])

    def test_development_population_and_missing_months(self):
        data=json.loads((ROOT/'web/app/portal-data.json').read_text(encoding='utf-8'))
        self.assertEqual(data['classes'],[{'horizon':0,'negative':188,'positive':171},{'horizon':7,'negative':157,'positive':156}])
        self.assertEqual(sum((m['negative'] or 0)+(m['positive'] or 0) for m in data['coverage']),359)
        missing=[m['month'] for m in data['coverage'] if m['negative'] is None]
        self.assertEqual(missing,[f'2025-{m:02d}' for m in range(2,8)])
        corr=pd.read_csv(ROOT/'results/eda/correlaciones_desarrollo_h0_14d.csv',index_col=0)
        columns=[corr.columns[i] for i in [0,3,6,7,8,9]]
        self.assertEqual(data['correlations'],corr.loc[columns,columns].values.tolist())
