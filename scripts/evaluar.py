"""Evaluación de modelos congelados en el período histórico aprobado."""
from pathlib import Path
import argparse
import json
from datetime import datetime, timezone
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score,
    precision_score, recall_score, roc_auc_score, average_precision_score,
    confusion_matrix, precision_recall_curve, roc_curve,
)
from procesar_datos import ROOT, FEATURES, require, sha, write_json, configured

def metrics(y, probability, prediction=None):
    y = np.asarray(y, dtype=int)
    probability = np.asarray(probability, dtype=float)
    pred = np.asarray(prediction, dtype=int) if prediction is not None else (probability >= .5).astype(int)
    both = len(np.unique(y)) == 2
    cm = confusion_matrix(y, pred, labels=[0, 1])
    normalized = np.divide(cm, cm.sum(axis=1, keepdims=True),
                           out=np.zeros((2, 2), dtype=float), where=cm.sum(axis=1, keepdims=True) != 0)
    return dict(f1=float(f1_score(y, pred, zero_division=0)),
                precision=float(precision_score(y, pred, zero_division=0)),
                recall=float(recall_score(y, pred, zero_division=0)),
                accuracy=float(accuracy_score(y, pred)),
                balanced_accuracy=float(balanced_accuracy_score(y, pred)) if both else None,
                roc_auc=float(roc_auc_score(y, probability)) if both else None,
                average_precision=float(average_precision_score(y, probability)) if both else None,
                motivo_indefinidas=None if both else 'una sola clase real',
                sin_predicciones_positivas=bool(not pred.any()),
                tn=int(cm[0, 0]), fp=int(cm[0, 1]), fn=int(cm[1, 0]), tp=int(cm[1, 1]),
                confusion_normalizada=normalized.tolist())


def majority(y_train, size):
    prevalence = float(np.mean(y_train))
    # En empate elige 0 aunque la puntuación constante sea 0.5.
    return np.full(size, prevalence), np.full(size, int(prevalence > .5))


def evaluate_record(rows, prediction_rows, name, h, window, role, fold, frame, probabilities, predictions=None):
    hard = np.asarray(predictions) if predictions is not None else (np.asarray(probabilities) >= .5).astype(int)
    rows.append(dict(config=name, horizonte=h, ventana=window, rol=role, fold=str(fold),
                     n=len(frame), **metrics(frame.y, probabilities, hard)))
    for (_, row), p, pred in zip(frame.iterrows(), probabilities, hard):
        prediction_rows.append(dict(config=name, horizonte=h, ventana=window, rol=role, fold=str(fold),
            muestra_id=row.muestra_id, fecha_referencia=row.fecha_referencia, fecha_objetivo=row.fecha_objetivo,
            resultado_t=int(row.resultado_t), y=int(row.y), probabilidad=float(p), prediccion=int(pred)))


def evaluate(data_dir, output):
    frozen = json.loads((output/'seleccion.json').read_text(encoding='utf-8'))
    require(sha(data_dir/'manifiesto.json') == frozen['input_hashes']['manifiesto.json'], 'Datos distintos de selección')
    selected = frozen['selected']
    partitions = json.loads((data_dir/'particiones.json').read_text(encoding='utf-8'))
    frames = {(h,w): pd.read_csv(data_dir/f'muestras_h{h}_{w}d.csv').set_index('muestra_id', drop=False)
              for h in [0,7] for w in [7,14]}
    metrics_file = output/'metricas.csv'
    prediction_file = output/'predictions/predicciones.csv'
    stored = pd.read_csv(metrics_file, dtype={'fold': str})
    predictions_frame = pd.read_csv(prediction_file, dtype={'fold': str})
    require(not stored.rol.eq('historico').any(), 'Esta ejecución ya fue evaluada')
    rows, predictions = stored.to_dict('records'), predictions_frame.to_dict('records')
    fitted = {}
    for record in frozen['configs']:
        name=record['config']
        model=joblib.load(output.parent/'models'/f'{name}.joblib')
        require(model['seleccion_sha256']==sha(output/'seleccion.json'), 'Modelo ajeno a selección')
        fitted[name]=(model['pipeline'],record['horizonte'],record['ventana'])
    timeline=json.loads((output/'secuencia.json').read_text(encoding='utf-8'))
    frozen_hash=sha(output/'seleccion.json')
    # Solo después de persistir selección se accede a las etiquetas históricas para métricas.
    for name, (model, horizon, window) in fitted.items():
        historical = partitions[str(horizon)]['historico']
        frame = frames[(horizon, window)]
        test = frame.loc[historical['validation_ids']]
        evaluate_record(rows, predictions, name, horizon, window, 'historico', 'historico',
                        test, model.predict_proba(test[FEATURES])[:, 1])
    for horizon in [0, 7]:
        fold = partitions[str(horizon)]['historico']
        frame = frames[(horizon, 7)]
        train, test = frame.loc[fold['train_ids']], frame.loc[fold['validation_ids']]
        p, hard = majority(train.y, len(test))
        evaluate_record(rows, predictions, f'mayoria_h{horizon}', horizon, 0,
                        'historico', 'historico', test, p, hard)
        if horizon == 7:
            evaluate_record(rows, predictions, 'persistencia_h7', 7, 0, 'historico', 'historico',
                            test, test.resultado_t, test.resultado_t)
    require(sha(output/'seleccion.json') == frozen_hash, 'Selección modificada durante evaluación')
    timeline.append({'evento': 'evaluacion_historica_completada', 'utc': datetime.now(timezone.utc).isoformat(),
                     'seleccion_sha256': frozen_hash})
    write_json(output/'secuencia.json', timeline)
    metrics_frame = pd.DataFrame(rows)
    metrics_frame.to_csv(output/'metricas.csv', index=False)
    pd.DataFrame(predictions).to_csv(output/'predictions/predicciones.csv', index=False)

    comparison = []
    for name, group in metrics_frame.groupby('config', sort=True):
        cv = group[group.rol.eq('validacion')]
        hist = group[group.rol.eq('historico')].iloc[0]
        row = dict(config=name, horizonte=int(hist.horizonte), ventana=int(hist.ventana),
                   seleccionado=selected[str(int(hist.horizonte))] == name,
                   cv_f1=float(cv.f1.mean()), cv_std=float(cv.f1.std(ddof=0)))
        for key in ['n', 'f1', 'precision', 'recall', 'accuracy', 'balanced_accuracy',
                    'roc_auc', 'average_precision', 'tn', 'fp', 'fn', 'tp']:
            row['historico_'+key] = float(hist[key])
        comparison.append(row)
    pd.DataFrame(comparison).to_csv(output/'comparacion.csv', index=False)

    return selected




def exportar_academicos(results_dir=None, data_dir=None):
    """Regenera curvas ROC/PR y particiones para el informe y el portal.

    No entrena modelos ni modifica las métricas oficiales.
    """
    results_dir = Path(results_dir) if results_dir is not None else configured('resultados')
    data_dir = Path(data_dir) if data_dir is not None else configured('datos')
    output = results_dir / 'academicos'
    output.mkdir(parents=True, exist_ok=True)

    source = results_dir / 'predictions/predicciones.csv'
    require(source.is_file(),
            'Falta results/predictions/predicciones.csv; no está incluido en el ZIP original. '
            'Regenerar predicciones con una ejecución reproducible antes de exportar curvas.')
    predictions = pd.read_csv(source)
    comparison = pd.read_csv(results_dir / 'comparacion.csv').set_index('config')
    curves = []
    for name, group in predictions[predictions.rol.eq('historico')].groupby('config'):
        cm = confusion_matrix(group.y, group.prediccion, labels=[0, 1]).ravel()
        expected = comparison.loc[
            name, ['historico_tn', 'historico_fp', 'historico_fn', 'historico_tp']
        ].tolist()
        require(list(cm) == expected, 'Matriz de confusión distinta')
        fpr, tpr, _ = roc_curve(group.y, group.probabilidad)
        precision, recall, _ = precision_recall_curve(group.y, group.probabilidad)
        curves.append({
            'config': name,
            'horizonte': int(group.horizonte.iloc[0]),
            'fpr': fpr.tolist(),
            'tpr': tpr.tolist(),
            'precision': precision.tolist(),
            'recall': recall.tolist(),
        })

    write_json(output / 'curvas.json', curves)
    summary = data_dir / 'resumen_particiones.csv'
    pd.read_csv(summary).to_csv(output / 'particiones.csv', index=False)
    write_json(output / 'manifiesto.json', {
        'origen_predicciones_sha256': sha(source),
        'origen_particiones_sha256': sha(summary),
        'seleccion_sha256': sha(results_dir / 'seleccion.json'),
        'script_sha256': sha(__file__),
        'contenido': 'Coordenadas de curvas agregadas; sin IDs, fechas individuales ni puntuaciones por muestra',
        'salidas': {
            path.name: sha(path)
            for path in [output / 'curvas.json', output / 'particiones.csv']
        },
    })
    return output


def exportar_portal(results_dir=None, portal_path=None):
    """Genera el JSON agregado consumido por el portal, sin datos individuales."""
    results = Path(results_dir) if results_dir is not None else configured('resultados')
    portal_path = Path(portal_path) if portal_path is not None else ROOT / 'web/app/portal-data.json'
    selected = json.loads((results / 'seleccion.json').read_text(encoding='utf-8'))['selected']
    comparison = pd.read_csv(results / 'comparacion.csv')

    models = []
    for row in comparison.to_dict('records'):
        models.append({
            'id': row['config'], 'horizon': row['horizonte'], 'window': row['ventana'],
            'selected': selected[str(row['horizonte'])] == row['config'],
            'cv': row['cv_f1'], 'f1': row['historico_f1'],
            'precision': row['historico_precision'], 'recall': row['historico_recall'],
            'n': int(row['historico_n']),
            'matrix': [[int(row['historico_tn']), int(row['historico_fp'])],
                       [int(row['historico_fn']), int(row['historico_tp'])]],
        })
    classes = []
    for horizon in [0, 7]:
        counts = pd.read_csv(results / f'eda/eda_clases_h{horizon}.csv').iloc[:, 1:].sum()
        classes.append({
            'horizon': horizon,
            'negative': int(counts.iloc[0]),
            'positive': int(counts.iloc[1]),
        })
    require([c['negative'] + c['positive'] for c in classes] == [359, 313],
            'Poblaciones de desarrollo distintas de las oficiales')

    monthly = pd.read_csv(results / 'eda/cobertura_mensual_desarrollo_h0.csv').set_index('mes')
    months = pd.period_range(monthly.index.min(), monthly.index.max(), freq='M')
    coverage = [{
        'month': str(month),
        'negative': int(monthly.loc[str(month)].iloc[0]) if str(month) in monthly.index else None,
        'positive': int(monthly.loc[str(month)].iloc[1]) if str(month) in monthly.index else None,
    } for month in months]
    require(sum((row['negative'] or 0) + (row['positive'] or 0) for row in coverage) == 359,
            'Cobertura mensual distinta de la oficial')

    variables = [FEATURES[i] for i in [0, 3, 6, 7, 8, 9]]
    corr = pd.read_csv(results / 'eda/correlaciones_desarrollo_h0_14d.csv', index_col=0)
    keys = [feature + '_14d' for feature in variables]
    payload = {
        'models': models,
        'classes': classes,
        'coverage': coverage,
        'correlations': corr.loc[keys, keys].values.tolist(),
    }
    portal_path.parent.mkdir(parents=True, exist_ok=True)
    portal_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + '\n',
                           encoding='utf-8')
    return portal_path

if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=configured('datos'))
    parser.add_argument('--results-dir', type=Path, default=configured('resultados'))
    args=parser.parse_args()
    print(evaluate(args.data_dir,args.results_dir))
