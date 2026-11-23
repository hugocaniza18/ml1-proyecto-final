"""Pruebas sintéticas de ajuste, referencias y selección de Etapa D."""
from pathlib import Path
import sys
import unittest
import json
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from evaluar import majority, metrics, evaluate_record
from entrenar import choose, estimator_grid
from procesar_datos import ROOT, FEATURES


class ExperimentTests(unittest.TestCase):
    def test_persistence_predictions_depend_only_on_current_state(self):
        frame = pd.DataFrame(dict(muestra_id=['a', 'b'], y=[1, 0], resultado_t=[0, 1],
                                  fecha_referencia=['2025-09-01']*2,
                                  fecha_objetivo=['2025-09-08']*2))
        rows, predictions = [], []
        evaluate_record(rows, predictions, 'persistencia_h7', 7, 0, 'validacion', 1,
                        frame, frame.resultado_t, frame.resultado_t)
        self.assertEqual([p['prediccion'] for p in predictions], [0, 1])
        frame['y'] = [0, 1]
        changed_rows, changed_predictions = [], []
        evaluate_record(changed_rows, changed_predictions, 'persistencia_h7', 7, 0, 'validacion', 1,
                        frame, frame.resultado_t, frame.resultado_t)
        self.assertEqual([p['prediccion'] for p in changed_predictions], [0, 1])

    def test_majority_uses_training_and_tie_predicts_zero(self):
        probability, hard = majority([0, 1], 3)
        np.testing.assert_array_equal(hard, [0, 0, 0])
        np.testing.assert_allclose(probability, .5)
        _, hard = majority([0, 1, 1], 2)
        np.testing.assert_array_equal(hard, [1, 1])

    def test_scaler_only_fits_training_and_feature_contract(self):
        config = json.loads((ROOT/'scripts/proyecto.json').read_text())['experimento']
        estimator, _ = estimator_grid('LR', config)
        train = pd.DataFrame(np.arange(60).reshape(6, 10), columns=FEATURES)
        estimator.fit(train, [0, 1, 0, 1, 0, 1])
        np.testing.assert_allclose(estimator['scale'].mean_, train.mean())
        before = estimator['scale'].mean_.copy()
        estimator.predict_proba(train + 10000)
        np.testing.assert_array_equal(estimator['scale'].mean_, before)
        self.assertEqual(list(estimator.feature_names_in_), FEATURES)
        with self.assertRaises(ValueError):
            estimator.predict(train.assign(resultado_t=0))

    def test_selection_ties_and_independence_from_historical_metrics(self):
        records = [dict(config='RF14', cv_f1=.6001, ventana=14, modelo='RF', historico_f1=1),
                   dict(config='RF7', cv_f1=.6002, ventana=7, modelo='RF', historico_f1=0),
                   dict(config='LR7', cv_f1=.6000, ventana=7, modelo='LR', historico_f1=0)]
        self.assertEqual(choose(records), 'LR7')
        for row in records:
            row['historico_f1'] = 1 - row['historico_f1']
        self.assertEqual(choose(records), 'LR7')

    def test_metrics_zero_predictions_and_single_class(self):
        record = metrics([0, 1], [0, 0])
        self.assertEqual(record['f1'], 0)
        self.assertTrue(record['sin_predicciones_positivas'])
        single = metrics([0, 0], [.1, .2])
        self.assertIsNone(single['roc_auc'])
        self.assertIsNone(single['average_precision'])
        self.assertIsNone(single['balanced_accuracy'])

    def test_grid_has_approved_order(self):
        config = json.loads((ROOT/'scripts/proyecto.json').read_text())['experimento']
        _, grid = estimator_grid('RF', config)
        self.assertEqual(len(grid), 9)
        self.assertEqual(grid[0], {'model__max_depth': [3], 'model__min_samples_leaf': [3]})
        self.assertEqual(grid[-1], {'model__max_depth': [None], 'model__min_samples_leaf': [10]})


if __name__ == '__main__':
    unittest.main()
