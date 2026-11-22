"""Verifica predicciones, métricas, modelos y repetición de Etapa D."""
from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd
import joblib
from evaluar import metrics, majority
from entrenar import choose, input_hashes, validate_inputs
from procesar_datos import ROOT, FEATURES, require, sha, write_json


def verify(data_dir, output, repeat=None):
    frozen_path = output/'seleccion.json'
    frozen = json.loads(frozen_path.read_text(encoding='utf-8'))
    require(frozen['input_hashes'] == input_hashes(data_dir), 'Hashes distintos a selección congelada')
    partitions = json.loads((data_dir/'particiones.json').read_text(encoding='utf-8'))
    frames = validate_inputs(data_dir, partitions)
    pred = pd.read_csv(output/'predictions/predicciones.csv', dtype={'fold': str})
    stored = pd.read_csv(output/'metricas.csv', dtype={'fold': str})
    require(len(stored) == 44, 'Se esperan 11 configuraciones/referencias × cuatro bloques')
    require(not pred.duplicated(['config', 'rol', 'fold', 'muestra_id']).any(), 'Predicciones duplicadas')
    timeline = json.loads((output/'secuencia.json').read_text(encoding='utf-8'))
    require([x['evento'] for x in timeline] == ['seleccion_congelada', 'evaluacion_historica_completada'],
            'Secuencia de evaluación incorrecta')
    require(timeline[0]['utc'] < timeline[1]['utc'], 'Tiempos de congelación incorrectos')
    require(timeline[0]['sha256'] == timeline[1]['seleccion_sha256'] == sha(frozen_path), 'Selección alterada')
    for row in stored.itertuples():
        group = pred[(pred.config == row.config) & (pred.rol == row.rol) & (pred.fold == row.fold)]
        frame = frames[(row.horizonte, row.ventana or 7)]
        block = partitions[str(row.horizonte)][row.fold]
        expected = frame.loc[block['validation_ids']]
        require(set(group.muestra_id) == set(expected.index), 'Población de evaluación distinta')
        expected = expected.loc[group.muestra_id]
        require(np.array_equal(expected.y, group.y), 'Etiquetas distintas')
        require(np.array_equal(expected.resultado_t, group.resultado_t), 'Estado en t distinto')
        require(np.isfinite(group.probabilidad).all() and group.probabilidad.between(0, 1).all(), 'Probabilidades inválidas')
        if row.config.startswith('mayoria'):
            p, hard = majority(frame.loc[block['train_ids']].y, len(group))
        elif row.config == 'persistencia_h7':
            require(row.horizonte == 7, 'Persistencia fuera de h7')
            p, hard = expected.resultado_t, expected.resultado_t
        else:
            p, hard = group.probabilidad, group.probabilidad.ge(.5).astype(int)
        require(np.allclose(p, group.probabilidad, atol=1e-12, rtol=0), 'Referencia distinta')
        require(np.array_equal(hard, group.prediccion), 'Clases distintas de regla')
        calculated = metrics(group.y, group.probabilidad, group.prediccion)
        for name in ['f1', 'precision', 'recall', 'accuracy', 'balanced_accuracy',
                     'roc_auc', 'average_precision', 'tn', 'fp', 'fn', 'tp']:
            require(np.isclose(getattr(row, name), calculated[name], atol=1e-12), f'Métrica distinta: {name}')
    comparison = pd.read_csv(output/'comparacion.csv')
    for horizon in [0, 7]:
        require(choose([r for r in frozen['configs'] if r['horizonte'] == horizon]) ==
                frozen['selected'][str(horizon)], 'Selección distinta de CV')
    max_error = 0.0
    require(len(list((output.parent/'models').glob('*.joblib'))) == 8, 'Cantidad de modelos distinta')
    for record in frozen['configs']:
        name = record['config']
        cv = stored[(stored.config == name) & (stored.rol == 'validacion')]
        require(np.isclose(cv.f1.mean(), record['cv_f1'], atol=1e-12), 'CV no coincide con predicciones')
        require(np.isclose(cv.f1.std(ddof=0), record['cv_std'], atol=1e-12), 'Desviación CV distinta')
        grid = pd.read_csv(output/'busquedas'/f'{name}.csv')
        require(np.isclose(grid.mean_test_score.max(), record['cv_f1'], atol=1e-12), 'Grilla distinta')
        saved = joblib.load(output.parent/'models'/f'{name}.joblib')
        require(saved['features'] == FEATURES and saved['threshold'] == .5, 'Contrato de modelo distinto')
        require(saved['seleccion_sha256'] == sha(frozen_path), 'Modelo ligado a otra selección')
        frame = frames[(record['horizonte'], record['ventana'])]
        group = pred[(pred.config == name) & (pred.rol == 'historico')]
        require(list(saved['pipeline'].feature_names_in_) == FEATURES, 'Features adicionales en pipeline')
        values = saved['pipeline'].predict_proba(frame.loc[group.muestra_id, FEATURES])[:, 1]
        max_error = max(max_error, float(np.max(np.abs(values-group.probabilidad))))
        if record['modelo'] == 'LR':
            train = frame.loc[partitions[str(record['horizonte'])]['historico']['train_ids']]
            require(np.allclose(saved['pipeline']['scale'].mean_, train[FEATURES].mean(), atol=1e-12),
                    'Scaler ajeno al desarrollo')
    require(max_error <= 1e-12, 'Puntuaciones de modelos recargados distintas')
    for row in comparison.itertuples():
        group = stored[stored.config.eq(row.config)]
        require(np.isclose(row.cv_f1, group[group.rol.eq('validacion')].f1.mean(), atol=1e-12),
                'Comparador CV inconsistente')
        historical = group[group.rol.eq('historico')].iloc[0]
        for metric in ['n', 'f1', 'precision', 'recall', 'accuracy', 'balanced_accuracy', 'roc_auc', 'average_precision']:
            require(np.isclose(getattr(row, 'historico_'+metric), historical[metric], atol=1e-12),
                    'Comparador histórico inconsistente')
    repeated = None
    if repeat:
        require(json.loads((repeat/'seleccion.json').read_text(encoding='utf-8')) == frozen,
                'Selección o configuración de repetición distinta')
        for filename in ['predicciones.csv', 'metricas.csv', 'comparacion.csv', 'aprendizaje.csv']:
            a = pd.read_csv(output/('predictions/'+filename if filename=='predicciones.csv' else filename))
            b = pd.read_csv(repeat/('predictions/'+filename if filename=='predicciones.csv' else filename))
            pd.testing.assert_frame_equal(a, b, check_exact=False, atol=1e-12, rtol=0)
        repeated = True
    return dict(metricas_recalculadas=len(stored), predicciones_verificadas=len(pred),
                modelos_recargados=8, error_maximo_modelos=max_error, seleccion_exclusiva_cv=True,
                congelacion_previa_verificada=True, repeticion_completa_coincide=repeated,
                hashes_entrada_intactos=True, evaluacion='histórica previamente observada')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=Path, default=ROOT/'data/processed')
    parser.add_argument('--output-dir', type=Path, default=ROOT/'results')
    parser.add_argument('--repeat-dir', type=Path)
    args = parser.parse_args()
    result = verify(args.data_dir, args.output_dir, args.repeat_dir)
    print(json.dumps(result, ensure_ascii=False, indent=2))
