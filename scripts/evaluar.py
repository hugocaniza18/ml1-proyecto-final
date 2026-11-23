"""Evaluación de modelos congelados en el período histórico aprobado."""
from pathlib import Path
import argparse
import json
from datetime import datetime, timezone
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score,
    precision_score, recall_score, roc_auc_score, average_precision_score, confusion_matrix)
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


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=configured('datos'))
    parser.add_argument('--results-dir', type=Path, default=configured('resultados'))
    args=parser.parse_args()
    print(evaluate(args.data_dir,args.results_dir))
