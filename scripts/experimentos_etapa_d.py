"""Ajuste en desarrollo y evaluación histórica, con selección congelada en disco."""
from pathlib import Path
import argparse
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score,
    precision_score, recall_score, roc_auc_score, average_precision_score, confusion_matrix)
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from procesar_horizontes import ROOT, FEATURES, sha, require, write_json, check_separation, verify_samples


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


def choose(records):
    return sorted(records, key=lambda r: (-round(r['cv_f1'], 3), r['ventana'],
                                          0 if r['modelo'] == 'LR' else 1))[0]['config']


def estimator_grid(model, config):
    if model == 'LR':
        estimator = Pipeline([('scale', StandardScaler()),
                              ('model', LogisticRegression(max_iter=config['lr_max_iter'],
                                                           random_state=config['seed']))])
        grid = [{'model__C': [c]} for c in config['lr_c']]
    else:
        estimator = Pipeline([('model', RandomForestClassifier(
            n_estimators=config['rf_n_estimators'], random_state=config['seed'], n_jobs=1))])
        grid = [{'model__max_depth': [d], 'model__min_samples_leaf': [m]}
                for d in config['rf_max_depth'] for m in config['rf_min_samples_leaf']]
    return estimator, grid


def input_hashes(data_dir):
    require(sha(data_dir/'manifiesto.json') == sha(ROOT/'reports/etapa_c/manifiesto_procesamiento.json'),
            'Manifiesto distinto del cierre de C')
    manifest = json.loads((data_dir/'manifiesto.json').read_text(encoding='utf-8'))
    for filename, digest in manifest['salidas'].items():
        require(sha(data_dir/filename) == digest, f'Entrada alterada: {filename}')
    require(manifest['version'] == 'horizontes-v2-etapa-c', 'Versión C inesperada')
    return {**manifest['salidas'], 'manifiesto.json': sha(data_dir/'manifiesto.json')}


def validate_trace(data_dir):
    path = ROOT/'data/processed/traza_era5_c02'
    trace = json.loads((path/'traza.json').read_text(encoding='utf-8'))
    require(trace['resumen']['muestras_sha256'] == sha(data_dir/'muestras.csv'), 'Traza de otras muestras')
    for name, digest in trace['salidas'].items():
        require(sha(path/name) == digest, 'Producto de traza alterado')
    hourly_root = ROOT.parent/'outputs/roya-soja/data/raw'
    for record in trace['respuestas'] + trace['manifiestos']:
        require(sha(hourly_root/record['archivo']) == record['sha256'], 'Respaldo ERA5 alterado')
    return sha(path/'traza.json')


def validate_inputs(data_dir, partitions):
    frames = {}
    features = json.loads((data_dir/'predictores.json').read_text(encoding='utf-8'))['features']
    require(features == FEATURES, 'Lista de predictores distinta')
    all_samples = pd.read_csv(data_dir/'muestras.csv')
    verify_samples(all_samples)
    expected = {0: [(164, 38), (188, 55), (239, 100)],
                7: [(141, 31), (150, 51), (196, 90)]}
    for horizon in [0, 7]:
        for window in [7, 14]:
            frame = pd.read_csv(data_dir/f'muestras_h{horizon}_{window}d.csv').set_index('muestra_id', drop=False)
            require(frame.index.is_unique, 'IDs duplicados')
            for number, sizes in enumerate(expected[horizon], 1):
                fold = partitions[str(horizon)][str(number)]
                tr, va = frame.loc[fold['train_ids']], frame.loc[fold['validation_ids']]
                require((len(tr), len(va)) == sizes, 'Fold distinto del aprobado')
                check_separation(tr, va)
                dates = sorted(pd.to_datetime(va.fecha_objetivo).unique())
                require(list(pd.to_datetime(dates)) == list(pd.date_range(
                    ['2025-08-18', '2025-09-29', '2025-11-10'][number-1], periods=6, freq='7D')),
                    'Fechas de validación distintas')
            historical = partitions[str(horizon)]['historico']
            tr, va = frame.loc[historical['train_ids']], frame.loc[historical['validation_ids']]
            require((len(tr), len(va)) == ((359, 66) if horizon == 0 else (313, 46)), 'Histórico distinto')
            check_separation(tr, va)
            require(pd.to_datetime(va.fecha_objetivo).between('2026-01-01', '2026-03-31').all(),
                    'Etiquetas fuera de evaluación histórica')
            require(set(tr.index) | set(va.index) | set(historical['purged_ids']) == set(frame.index),
                    'Partición histórica incompleta')
            frames[(horizon, window)] = frame
    return frames


def evaluate_record(rows, prediction_rows, name, h, window, role, fold, frame, probabilities, predictions=None):
    hard = np.asarray(predictions) if predictions is not None else (np.asarray(probabilities) >= .5).astype(int)
    rows.append(dict(config=name, horizonte=h, ventana=window, rol=role, fold=str(fold),
                     n=len(frame), **metrics(frame.y, probabilities, hard)))
    for (_, row), p, pred in zip(frame.iterrows(), probabilities, hard):
        prediction_rows.append(dict(config=name, horizonte=h, ventana=window, rol=role, fold=str(fold),
            muestra_id=row.muestra_id, fecha_referencia=row.fecha_referencia, fecha_objetivo=row.fecha_objetivo,
            resultado_t=int(row.resultado_t), y=int(row.y), probabilidad=float(p), prediccion=int(pred)))


def run(data_dir, output, config_path):
    require(not output.exists(), 'Destino existente; elegir una ejecución nueva')
    config = json.loads(config_path.read_text(encoding='utf-8'))
    approved = json.loads((ROOT/'config/etapa_d.json').read_text(encoding='utf-8'))
    require(config == approved, 'Configuración distinta de la aprobada')
    require(sklearn.__version__ == '1.6.1', 'Usar scikit-learn 1.6.1 del entorno fijado')
    hashes = input_hashes(data_dir)
    trace_hash = validate_trace(data_dir)
    partitions = json.loads((data_dir/'particiones.json').read_text(encoding='utf-8'))
    frames = validate_inputs(data_dir, partitions)
    output.mkdir(parents=True)
    (output/'models').mkdir()
    (output/'grids').mkdir()
    write_json(output/'config.json', config)
    rows, predictions, learning, configs, fitted = [], [], [], [], {}
    timeline = []
    for horizon in [0, 7]:
        for window in [7, 14]:
            frame = frames[(horizon, window)]
            development = frame.loc[partitions[str(horizon)]['historico']['train_ids']]
            y, x = development.y, development[FEATURES]
            cv = []
            for number in [1, 2, 3]:
                fold = partitions[str(horizon)][str(number)]
                tr = development.index.get_indexer(fold['train_ids'])
                va = development.index.get_indexer(fold['validation_ids'])
                require((tr >= 0).all() and (va >= 0).all(), 'IDs de fold fuera de desarrollo')
                cv.append((tr, va))
            for model in ['LR', 'RF']:
                name = f'{model}_{window}d_h{horizon}'
                estimator, grid = estimator_grid(model, config)
                search = GridSearchCV(estimator, grid, cv=cv, scoring='f1', refit=True,
                                      n_jobs=2, error_score='raise', return_train_score=True)
                search.fit(x, y)
                grid_rows = pd.DataFrame(search.cv_results_)
                grid_rows.to_csv(output/'grids'/f'{name}.csv', index=False)
                record = dict(config=name, horizonte=horizon, ventana=window, modelo=model,
                              cv_f1=float(search.best_score_),
                              cv_std=float(search.cv_results_['std_test_score'][search.best_index_]),
                              parametros=search.best_params_)
                configs.append(record)
                fitted[name] = (search.best_estimator_, horizon, window)
                for number, (tr, va) in enumerate(cv, 1):
                    est = clone(search.best_estimator_).fit(x.iloc[tr], y.iloc[tr])
                    valid = development.iloc[va]
                    evaluate_record(rows, predictions, name, horizon, window, 'validacion', number,
                                    valid, est.predict_proba(x.iloc[va])[:, 1])
                    dates = sorted(development.iloc[tr].fecha_referencia.unique())
                    for fraction in config['learning_fractions']:
                        chosen = dates[:max(1, int(np.ceil(len(dates)*fraction)))]
                        prefix = development.iloc[tr][development.iloc[tr].fecha_referencia.isin(chosen)]
                        if prefix.y.nunique() < 2:
                            learning.append(dict(config=name, horizonte=horizon, fold=number,
                                fraccion=fraction, n=len(prefix), estado='omitido_una_clase'))
                            continue
                        part = clone(search.best_estimator_).fit(prefix[FEATURES], prefix.y)
                        learning.append(dict(config=name, horizonte=horizon, fold=number,
                            fraccion=fraction, n=len(prefix), estado='ok',
                            train_f1=float(f1_score(prefix.y, part.predict(prefix[FEATURES]), zero_division=0)),
                            validation_f1=float(f1_score(valid.y, part.predict(valid[FEATURES]), zero_division=0))))
                print(f"{name}: CV F1={record['cv_f1']:.6f}", flush=True)
    # Las referencias se ajustan con los mismos entrenamientos.
    for horizon in [0, 7]:
        frame = frames[(horizon, 7)]
        for number in [1, 2, 3]:
            fold = partitions[str(horizon)][str(number)]
            train, valid = frame.loc[fold['train_ids']], frame.loc[fold['validation_ids']]
            p, hard = majority(train.y, len(valid))
            evaluate_record(rows, predictions, f'mayoria_h{horizon}', horizon, 0,
                            'validacion', number, valid, p, hard)
            if horizon == 7:
                evaluate_record(rows, predictions, 'persistencia_h7', 7, 0, 'validacion',
                                number, valid, valid.resultado_t, valid.resultado_t)
    selected = {str(h): choose([r for r in configs if r['horizonte'] == h]) for h in [0, 7]}
    frozen = dict(selected=selected, configs=configs, input_hashes=hashes, traza_sha256=trace_hash, config=config,
                  config_sha256=sha(config_path))
    write_json(output/'seleccion_congelada.json', frozen)
    frozen_hash = sha(output/'seleccion_congelada.json')
    timeline.append({'evento': 'seleccion_congelada', 'utc': datetime.now(timezone.utc).isoformat(),
                     'sha256': frozen_hash})
    write_json(output/'secuencia.json', timeline)
    # Solo después de persistir selección se accede a las etiquetas históricas para métricas.
    for name, (model, horizon, window) in fitted.items():
        historical = partitions[str(horizon)]['historico']
        frame = frames[(horizon, window)]
        test = frame.loc[historical['validation_ids']]
        joblib.dump(dict(pipeline=model, features=FEATURES, horizonte=horizon, ventana=window,
                         threshold=.5, input_hashes=hashes, seleccion_sha256=frozen_hash),
                    output/'models'/f'{name}.joblib')
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
    require(sha(output/'seleccion_congelada.json') == frozen_hash, 'Selección modificada durante evaluación')
    timeline.append({'evento': 'evaluacion_historica_completada', 'utc': datetime.now(timezone.utc).isoformat(),
                     'seleccion_sha256': frozen_hash})
    write_json(output/'secuencia.json', timeline)
    metrics_frame = pd.DataFrame(rows)
    metrics_frame.to_csv(output/'metricas.csv', index=False)
    pd.DataFrame(predictions).to_csv(output/'predicciones.csv', index=False)
    pd.DataFrame(learning).to_csv(output/'aprendizaje.csv', index=False)
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
    require(input_hashes(data_dir) == hashes, 'Datos cambiaron durante el experimento')
    try:
        commit = subprocess.check_output(['git', '-c', f'safe.directory={ROOT.as_posix()}',
                                         'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    except subprocess.CalledProcessError:
        commit = 'no_disponible'
    write_json(output/'entorno.json', dict(python=platform.python_version(), numpy=np.__version__,
        pandas=pd.__version__, sklearn=sklearn.__version__, joblib=joblib.__version__,
        commit_base=commit, codigo_sha256=sha(__file__),
        archivos_codigo={f: sha(ROOT/f) for f in ['scripts/procesar_horizontes.py',
                                                 'scripts/experimentos_etapa_d.py', 'config/etapa_d.json']},
        nota_commit='Código de D identificado por SHA-256 sobre worktree; commit base no incluye cambios no confirmados',
        evaluacion=config['evaluation'], disponibilidad='no_acreditada'))
    return selected


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=ROOT/'data/processed/horizontes_v2_c02')
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--config', type=Path, default=ROOT/'config/etapa_d.json')
    args = parser.parse_args()
    print(run(args.data_dir, args.output_dir, args.config))
