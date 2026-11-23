"""Ajuste en desarrollo y evaluación histórica, con selección congelada en disco."""
from pathlib import Path
import argparse
import json
import platform
from datetime import datetime, timezone
import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from procesar_datos import huellas, ROOT, FEATURES, sha, require, write_json, check_separation, verify_samples, configured, project


from evaluar import majority, evaluate_record

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
    manifest = json.loads((data_dir/'manifiesto.json').read_text(encoding='utf-8'))
    approved = huellas()['procesados']
    require(manifest['salidas'] == approved, 'Tablas distintas del cierre de C')
    for filename, digest in manifest['salidas'].items():
        require(sha(data_dir/filename) == digest, f'Entrada alterada: {filename}')
    require(manifest['version'] in ['horizontes-v2-etapa-c', 'horizontes-vigentes'], 'Versión C inesperada')
    return {**manifest['salidas'], 'manifiesto.json': sha(data_dir/'manifiesto.json')}


def validate_trace(data_dir):
    path = configured('intermedios')
    trace = json.loads((path/'traza.json').read_text(encoding='utf-8'))
    require(trace['resumen']['muestras_sha256'] == sha(data_dir/'muestras.csv'), 'Traza de otras muestras')
    for name, digest in trace['salidas'].items():
        require(sha(path/name) == digest, 'Producto de traza alterado')
    hourly_root = configured('era5')
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


def run(data_dir, output, config_path):
    require(not output.exists(), 'Destino existente; elegir una ejecución nueva')
    require(not (output.parent/'models').exists(), 'Modelos existentes; elegir una raíz de ejecución nueva')
    config = json.loads(config_path.read_text(encoding='utf-8'))['experimento']
    approved = project()['experimento']
    require(config == approved, 'Configuración distinta de la aprobada')
    require(sklearn.__version__ == '1.6.1', 'Usar scikit-learn 1.6.1 del entorno fijado')
    hashes = input_hashes(data_dir)
    trace_hash = validate_trace(data_dir)
    partitions = json.loads((data_dir/'particiones.json').read_text(encoding='utf-8'))
    frames = validate_inputs(data_dir, partitions)
    output.mkdir(parents=True)
    (output.parent/'models').mkdir()
    (output/'predictions').mkdir()
    (output/'busquedas').mkdir()

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
                grid_rows.to_csv(output/'busquedas'/f'{name}.csv', index=False)
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
    write_json(output/'seleccion.json', frozen)
    frozen_hash = sha(output/'seleccion.json')
    timeline.append({'evento': 'seleccion_congelada', 'utc': datetime.now(timezone.utc).isoformat(),
                     'sha256': frozen_hash})
    write_json(output/'secuencia.json', timeline)

    for name,(model,horizon,window) in fitted.items():
        joblib.dump(dict(pipeline=model,features=FEATURES,horizonte=horizon,ventana=window,
                         threshold=.5,input_hashes=hashes,seleccion_sha256=frozen_hash),
                    output.parent/'models'/f'{name}.joblib')
    pd.DataFrame(rows).to_csv(output/'metricas.csv',index=False)
    pd.DataFrame(predictions).to_csv(output/'predictions/predicciones.csv',index=False)
    pd.DataFrame(learning).to_csv(output/'aprendizaje.csv',index=False)
    require(input_hashes(data_dir)==hashes,'Datos cambiaron durante ajuste')
    write_json(output/'entorno.json',dict(python=platform.python_version(),numpy=np.__version__,
        pandas=pd.__version__,sklearn=sklearn.__version__,joblib=joblib.__version__,
        codigo_sha256=sha(__file__),config_sha256=sha(config_path)))
    return selected



# V2: experimento adicional, separado del cierre oficial V1.
def feature_groups_v2():
    from procesar_datos import FAVORABILITY_FEATURES
    return {
        'favorabilidad': list(FAVORABILITY_FEATURES),
        'combinadas': list(FEATURES) + list(FAVORABILITY_FEATURES),
    }


def prepare_v2(data_path, partitions_path):
    """Valida población y folds V2 contra las particiones oficiales de V1."""
    data_path = Path(data_path)
    partitions_path = Path(partitions_path)
    source_path = partitions_path.parent / 'muestras.csv'
    require(source_path.is_file(), 'Falta data/processed/muestras.csv de V1')
    require(data_path.is_file(), 'Ejecutar primero procesar_datos.py --version v2')
    frame = pd.read_csv(data_path, dtype={'ubicacion_id': str})
    original = pd.read_csv(source_path, dtype={'ubicacion_id': str})
    keys = ['muestra_id', 'ventana_dias']
    require(not frame.duplicated(keys).any(), 'Claves duplicadas en V2')
    require(len(frame) == len(original), 'V2 modificó el número de muestras')
    require(frame[keys + ['horizonte_dias', 'y', 'fecha_referencia', 'fecha_objetivo']].equals(
        original[keys + ['horizonte_dias', 'y', 'fecha_referencia', 'fecha_objetivo']]),
        'V2 modificó claves, etiquetas o fechas oficiales')
    groups = feature_groups_v2()
    for features in groups.values():
        require(all(f in frame.columns for f in features), 'Faltan predictores V2')
        values = frame[features].to_numpy(dtype=float)
        require(np.isfinite(values).all(), 'Predictores V2 nulos o no finitos')
    require(all(pd.to_datetime(frame['ventana_fin']) < pd.to_datetime(frame['fecha_referencia'])),
            'V2 incluye meteorología de la fecha de predicción o posterior')
    partitions = json.loads(partitions_path.read_text(encoding='utf-8'))
    frames = {}
    for h in [0, 7]:
        for w in [7, 14]:
            part = frame[(frame.horizonte_dias == h) & (frame.ventana_dias == w)].copy()
            part = part.set_index('muestra_id', drop=False)
            require(part.index.is_unique, 'ID repetido en horizonte/ventana V2')
            require(set(part.index) == set(original.loc[
                (original.horizonte_dias == h) & (original.ventana_dias == w), 'muestra_id']),
                'Población V2 distinta de V1')
            for fold_name in ['1', '2', '3', 'historico']:
                fold = partitions[str(h)][fold_name]
                tr, va = part.loc[fold['train_ids']], part.loc[fold['validation_ids']]
                check_separation(tr, va)
                require(not set(tr.index).intersection(va.index), 'Solapamiento de IDs')
            frames[(h, w)] = part
    return frames, partitions, groups


def run_v2(data_path, output, config_path, partitions_path):
    """Compara favorabilidad y combinadas con CV temporal; evalúa histórico al final."""
    output = Path(output)
    require(not output.exists(), 'Salida V2 existente: elegir otra ruta, no sobrescribir')
    require(not (output.parent/'models_v2').exists(), 'Modelos V2 existentes en esa raíz')
    config = json.loads(Path(config_path).read_text(encoding='utf-8'))['experimento']
    # Se conservan los hiperparámetros oficiales, sin exigir hashes de V1 al dataset V2.
    require(config == project()['experimento'], 'Configuración V1 modificada')
    frames, partitions, groups = prepare_v2(data_path, partitions_path)
    output.mkdir(parents=True)
    (output/'busquedas').mkdir()
    model_dir = output/'modelos'
    model_dir.mkdir()
    configs, metrics_rows, predictions = [], [], []
    selected = {}
    for group_name, features in groups.items():
        selected[group_name] = {}
        for horizon in [0, 7]:
            candidates = []
            for window in [7, 14]:
                frame = frames[(horizon, window)]
                historical = partitions[str(horizon)]['historico']
                dev = frame.loc[historical['train_ids']]
                folds = []
                for number in [1, 2, 3]:
                    fold = partitions[str(horizon)][str(number)]
                    tr = dev.index.get_indexer(fold['train_ids'])
                    va = dev.index.get_indexer(fold['validation_ids'])
                    require((tr >= 0).all() and (va >= 0).all(),
                            'Fold fuera del conjunto de desarrollo')
                    folds.append((tr, va))
                for model_name in ['LR', 'RF']:
                    name = f'{group_name}_{model_name}_{window}d_h{horizon}'
                    estimator, grid = estimator_grid(model_name, config)
                    search = GridSearchCV(estimator, grid, cv=folds, scoring='f1', refit=True,
                                          n_jobs=2, error_score='raise')
                    search.fit(dev[features], dev.y)
                    pd.DataFrame(search.cv_results_).to_csv(output/'busquedas'/f'{name}.csv', index=False)
                    record = {'grupo': group_name, 'config': name, 'horizonte': horizon,
                              'ventana': window, 'modelo': model_name, 'cv_f1': float(search.best_score_),
                              'cv_std': float(search.cv_results_['std_test_score'][search.best_index_]),
                              'parametros': search.best_params_}
                    configs.append(record)
                    candidates.append(record)
                    print(f"{name}: CV F1={record['cv_f1']:.4f}", flush=True)
            best_name = choose(candidates)
            best = next(record for record in candidates if record['config'] == best_name)
            selected[group_name][str(horizon)] = best_name
            frame = frames[(horizon, best['ventana'])]
            historical = partitions[str(horizon)]['historico']
            dev = frame.loc[historical['train_ids']]
            holdout = frame.loc[historical['validation_ids']]
            estimator, _ = estimator_grid(best['modelo'], config)
            estimator.set_params(**best['parametros'])
            estimator.fit(dev[features], dev.y)
            probabilities = estimator.predict_proba(holdout[features])[:, 1]
            evaluate_record(metrics_rows, predictions, best_name, horizon, best['ventana'],
                            'historico', 0, holdout, probabilities)
            joblib.dump({'pipeline': estimator, 'features': features, 'horizonte': horizon,
                         'ventana': best['ventana'], 'threshold': .5, 'grupo': group_name},
                        model_dir/f'{best_name}.joblib')
    pd.DataFrame(configs).to_csv(output/'configuraciones.csv', index=False)
    pd.DataFrame(metrics_rows).to_csv(output/'metricas.csv', index=False)
    pd.DataFrame(predictions).to_csv(output/'predicciones.csv', index=False)
    write_json(output/'seleccion.json', {'version': 'v2-experimental', 'seleccion': selected,
               'variables': groups, 'datos_sha256': sha(data_path),
               'particiones_sha256': sha(partitions_path), 'config_sha256': sha(config_path),
               'nota': 'Seleccion por CV temporal en desarrollo; historico solo evaluacion final'})
    return selected

if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir',type=Path,default=configured('datos'))
    parser.add_argument('--output-dir',type=Path,default=configured('resultados'))
    parser.add_argument('--config',type=Path,default=ROOT/'scripts/proyecto.json')
    parser.add_argument('--version', choices=['v1', 'v2'], default='v1')
    parser.add_argument('--v2-data', type=Path, default=ROOT/'data/processed/v2/muestras_v2.csv')
    parser.add_argument('--partitions', type=Path, default=ROOT/'data/processed/particiones.json')
    args=parser.parse_args()
    if args.version == 'v2':
        destination = args.output_dir if args.output_dir != configured('resultados') else ROOT/'results/v2'
        print(run_v2(args.v2_data, destination, args.config, args.partitions))
    else:
        print(run(args.data_dir,args.output_dir,args.config))
