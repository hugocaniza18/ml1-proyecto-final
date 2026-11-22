"""Etapa C offline: muestras y particiones aprobadas, sin ajuste de modelos."""
from pathlib import Path
import argparse
import hashlib
import json
import platform
from decimal import Decimal

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FEATURES = [
    'hr_media_pct', 'hr_min_pct', 'hr_max_pct',
    'temperatura_media_c', 'temperatura_min_c', 'temperatura_max_c',
    'precipitacion_mm', 'viento_vel_media_10m_ms',
    'viento_u_media_10m_ms', 'viento_v_media_10m_ms',
]
AGGREGATIONS = {
    feature: ('min' if '_min_' in feature else 'max' if '_max_' in feature
              else 'sum' if feature == 'precipitacion_mm' else 'mean')
    for feature in FEATURES
}
BLOCK_STARTS = ['2025-08-18', '2025-09-29', '2025-11-10']
LABELS = {'Negativo': 0, 'Positivo': 1}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def write_json(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2,
                                    allow_nan=False), encoding='utf-8')


def location_id(lat, lon):
    def canonical(value):
        number = Decimal(str(value))
        require(number.is_finite(), 'Coordenada no finita')
        if number == 0:
            return '0'
        text = format(number, 'f')
        return text.rstrip('0').rstrip('.') if '.' in text else text
    return f'{canonical(lat)},{canonical(lon)}'


def observation_id(location, date):
    return 'O' + hashlib.sha256(f'{location}|{date}'.encode()).hexdigest()[:20]


def observations(original):
    frame = original.copy()
    frame['fecha_registro'] = pd.to_datetime(frame.fecha_registro, errors='raise').dt.strftime('%Y-%m-%d')
    coordinates = frame[['latitud', 'longitud']].apply(pd.to_numeric, errors='raise')
    require(np.isfinite(coordinates).all().all(), 'Coordenadas no finitas')
    require(coordinates.latitud.between(-90, 90).all() and coordinates.longitud.between(-180, 180).all(),
            'Coordenadas fuera de rango')
    require(frame.clase_original.isin([*LABELS, 'Não Avaliado']).all(), 'Clase desconocida')
    codes = {79: 'Positivo', 80: 'Negativo', 82: 'Não Avaliado'}
    require(frame.clase_codigo.map(codes).eq(frame.clase_original).all(), 'Código y etiqueta incompatibles')
    frame['fila_fuente'] = np.arange(len(frame)) + 2
    frame['ubicacion_id'] = [location_id(a, b) for a, b in zip(frame.latitud, frame.longitud)]
    frame['observacion_id'] = [observation_id(a, b) for a, b in zip(frame.ubicacion_id, frame.fecha_registro)]
    frame['estado'] = 'evaluado'
    frame.loc[frame.clase_original.eq('Não Avaliado'), 'estado'] = 'no_evaluado'
    conflict = frame.duplicated(['ubicacion_id', 'fecha_registro'], keep=False)
    conflict |= frame.conflicto_clase_mismo_punto_fecha.ne(0)
    frame.loc[conflict, 'estado'] = 'conflicto'
    frame['disponibilidad'] = 'no_acreditada'
    return frame


def weather_windows(daily):
    """Agrega los diarios auditados; conserva la relación con id_medicion histórico."""
    required = ['id_medicion', 'fecha_objetivo', 'fecha_dato', 'dias_anteriores',
                'latitud', 'longitud', *FEATURES]
    data = daily[required].copy()  # Selección explícita: no procesa índices auxiliares.
    for col in ['fecha_objetivo', 'fecha_dato']:
        data[col] = pd.to_datetime(data[col], errors='raise')
    require(not data.duplicated(['id_medicion', 'fecha_dato']).any(), 'Duplicado meteorológico')
    require(data.groupby('id_medicion')[['latitud', 'longitud', 'fecha_objetivo']].nunique().le(1).all().all(),
            'Identificador meteorológico con referencias distintas')
    require((data.fecha_objetivo - data.fecha_dato).dt.days.eq(data.dias_anteriores).all(),
            'Fecha meteorológica incompatible con lag')
    require(data.dias_anteriores.isin(range(1, 15)).all(), 'Lag fuera de 1..14')
    data['ubicacion_id'] = [location_id(a, b) for a, b in zip(data.latitud, data.longitud)]
    require(not data.duplicated(['ubicacion_id', 'fecha_objetivo', 'dias_anteriores']).any(),
            'Referencias meteorológicas duplicadas')
    windows = {}
    excluded = {}
    for mid, group in data.groupby('id_medicion', sort=True):
        row = group.iloc[0]
        key = (row.ubicacion_id, row.fecha_objetivo.strftime('%Y-%m-%d'))
        if len(group) != 14 or set(group.dias_anteriores) != set(range(1, 15)):
            excluded[key] = 'cobertura_meteorologica_incompleta'
            continue
        if not np.isfinite(group[FEATURES].to_numpy()).all():
            excluded[key] = 'meteorologia_no_finita'
            continue
        require(group[['hr_min_pct', 'hr_media_pct', 'hr_max_pct']].apply(lambda x: x.between(0, 100)).all().all(),
                'HR fuera de rango')
        for prefix in ['hr', 'temperatura']:
            suffix = 'pct' if prefix == 'hr' else 'c'
            require((group[f'{prefix}_min_{suffix}'] <= group[f'{prefix}_media_{suffix}']).all()
                    and (group[f'{prefix}_media_{suffix}'] <= group[f'{prefix}_max_{suffix}']).all(),
                    'Extremos meteorológicos inconsistentes')
        require(group.precipitacion_mm.ge(0).all() and group.viento_vel_media_10m_ms.ge(0).all(),
                'Precipitación o velocidad negativa')
        require((np.hypot(group.viento_u_media_10m_ms, group.viento_v_media_10m_ms)
                 <= group.viento_vel_media_10m_ms + 1e-9).all(), 'Viento vectorial inválido')
        windows[key] = {
            w: {'id_medicion_anterior': str(mid), **group[group.dias_anteriores.le(w)]
                .agg(AGGREGATIONS).to_dict()} for w in [7, 14]
        }
    return windows, excluded


def make_samples(obs, weather, weather_exclusions):
    usable = obs[obs.estado.ne('conflicto')]
    lookup = {(r.ubicacion_id, r.fecha_registro): r for r in usable.itertuples()}
    conflicts = set(zip(obs.loc[obs.estado.eq('conflicto'), 'ubicacion_id'],
                        obs.loc[obs.estado.eq('conflicto'), 'fecha_registro']))
    samples, excluded = [], []
    for row in obs.itertuples():
        for horizon in [0, 7]:
            target_date = (pd.Timestamp(row.fecha_registro) + pd.Timedelta(days=horizon)).strftime('%Y-%m-%d')
            target_key = (row.ubicacion_id, target_date)
            key = (row.ubicacion_id, row.fecha_registro)
            target = lookup.get(target_key)
            reason = None
            if row.estado != 'evaluado':
                reason = 'referencia_' + row.estado
            elif target_key in conflicts:
                reason = 'objetivo_conflicto'
            elif target is None:
                reason = 'objetivo_sin_registro'
            elif target.estado != 'evaluado':
                reason = 'objetivo_' + target.estado
            elif key not in weather:
                reason = weather_exclusions.get(key, 'sin_meteorologia')
            if reason:
                excluded.append(dict(observacion_id=row.observacion_id, fila_fuente=row.fila_fuente,
                                     horizonte_dias=horizon, motivo=reason,
                                     fecha_referencia=row.fecha_registro, fecha_objetivo=target_date))
                continue
            for window in [7, 14]:
                values = weather[key][window]
                samples.append(dict(
                    muestra_id=f'{row.observacion_id}_h{horizon}', ubicacion_id=row.ubicacion_id,
                    observacion_id=row.observacion_id, observacion_objetivo_id=target.observacion_id,
                    fila_fuente=row.fila_fuente, fila_objetivo_fuente=target.fila_fuente,
                    punto_id=row.punto_id, punto_objetivo_id=target.punto_id,
                    latitud=row.latitud, longitud=row.longitud,
                    fecha_referencia=row.fecha_registro, fecha_objetivo=target_date,
                    horizonte_dias=horizon, ventana_dias=window,
                    ventana_inicio=(pd.Timestamp(row.fecha_registro)-pd.Timedelta(days=window)).strftime('%Y-%m-%d'),
                    ventana_fin=(pd.Timestamp(row.fecha_registro)-pd.Timedelta(days=1)).strftime('%Y-%m-%d'),
                    y=LABELS[target.clase_original], resultado_t=LABELS[row.clase_original],
                    disponibilidad='no_acreditada', fuente_etiqueta=row.fuente_url,
                    fuente_meteo='Open-Meteo ERA5; diario auditado Etapa B', **values))
    return pd.DataFrame(samples).sort_values(['horizonte_dias', 'fecha_referencia', 'muestra_id', 'ventana_dias']), pd.DataFrame(excluded)


def describe(frame):
    if frame.empty:
        return {'n': 0}
    ref = pd.to_datetime(frame.fecha_referencia)
    target = pd.to_datetime(frame.fecha_objetivo)
    return dict(n=len(frame), negativos=int(frame.y.eq(0).sum()), positivos=int(frame.y.eq(1).sum()),
                ubicaciones=frame.ubicacion_id.nunique(), punto_ids=frame.punto_id.nunique(),
                referencia_inicio=str(ref.min().date()), referencia_fin=str(ref.max().date()),
                etiqueta_inicio=str(target.min().date()), etiqueta_fin=str(target.max().date()),
                duracion_dias=int((ref.max()-ref.min()).days),
                dias_inclusivos=int((ref.max()-ref.min()).days)+1,
                fechas_observadas=ref.nunique(),
                meteo_inicio=str((ref.min()-pd.Timedelta(days=14)).date()),
                meteo_fin=str((ref.max()-pd.Timedelta(days=1)).date()),
                disponibilidad='no_acreditada')


def check_separation(train, validation):
    require(set(train.muestra_id).isdisjoint(validation.muestra_id), 'IDs compartidos entre particiones')
    require(train.y.nunique() == validation.y.nunique() == 2, 'Partición sin ambas clases')
    v = pd.to_datetime(validation.fecha_referencia).min()
    require(pd.to_datetime(train.fecha_objetivo).max() < v-pd.Timedelta(days=14),
            'Fuga temporal de etiquetas')
    require(pd.to_datetime(train.ventana_fin).max() < pd.to_datetime(validation.ventana_inicio).min(),
            'Solapamiento meteorológico')


def make_partitions(samples):
    folds, summary, assignments = {}, [], []
    # Una fila por muestra. La pertenencia se comparte entre ambas ventanas.
    for horizon in [0, 7]:
        population = samples[(samples.horizonte_dias == horizon) & (samples.ventana_dias == 14)].copy()
        target = pd.to_datetime(population.fecha_objetivo)
        ref = pd.to_datetime(population.fecha_referencia)
        test = population[target.between('2026-01-01', '2026-03-31')]
        require(not test.empty, 'Evaluación histórica vacía')
        v = pd.to_datetime(test.fecha_referencia).min()
        development = population[target < v-pd.Timedelta(days=14)]
        purge = population[(target < pd.Timestamp('2026-01-01')) & ~population.index.isin(development.index)]
        outside = population[~population.index.isin(development.index.union(test.index).union(purge.index))]
        require(outside.empty, 'Datos fuera del período aprobado: requieren nueva revisión')
        check_separation(development, test)
        blocks = [('historico', development, test, purge)]
        for number, start in enumerate(BLOCK_STARTS, 1):
            dates = pd.date_range(start, periods=6, freq='7D')
            valid = development[pd.to_datetime(development.fecha_objetivo).isin(dates)]
            require(set(pd.to_datetime(valid.fecha_objetivo)) == set(dates), 'Faltan fechas de fold aprobado')
            first_ref = pd.to_datetime(valid.fecha_referencia).min()
            fit = development[pd.to_datetime(development.fecha_objetivo) < first_ref-pd.Timedelta(days=14)]
            purged = development[(pd.to_datetime(development.fecha_referencia) < first_ref)
                                 & ~development.index.isin(fit.index)]
            check_separation(fit, valid)
            blocks.append((str(number), fit, valid, purged))
        folds[str(horizon)] = {}
        for name, train, valid, purged in blocks:
            valid_role = 'evaluacion_historica' if name == 'historico' else 'validacion'
            folds[str(horizon)][name] = {
                'train_ids': train.muestra_id.tolist(), 'validation_ids': valid.muestra_id.tolist(),
                'purged_ids': purged.muestra_id.tolist(),
                'fecha_inicio_referencia_validacion': valid.fecha_referencia.min(),
                'corte_etiqueta_train_exclusivo': str((pd.to_datetime(valid.fecha_referencia).min()
                                                     - pd.Timedelta(days=14)).date())}
            for role, frame in [('entrenamiento', train), (valid_role, valid), ('purga', purged)]:
                summary.append({'horizonte': horizon, 'bloque': name, 'rol': role, **describe(frame)})
                assignments.extend({'horizonte': horizon, 'bloque': name, 'rol': role, 'muestra_id': mid}
                                   for mid in frame.muestra_id)
    return folds, pd.DataFrame(summary), pd.DataFrame(assignments)


def verify_samples(samples):
    require(set(samples.ventana_dias) == {7, 14}, 'Ventanas incompletas')
    require(set(samples.horizonte_dias) == {0, 7}, 'Horizontes incompletos')
    require(not samples.duplicated(['muestra_id', 'ventana_dias']).any(), 'Muestras duplicadas')
    require(not samples.duplicated(['ubicacion_id', 'fecha_referencia', 'horizonte_dias', 'ventana_dias']).any(),
            'Clave de observación duplicada')
    require(not any('iith' in c.lower() for c in samples), 'Columna no autorizada')
    require(np.isfinite(samples[FEATURES].to_numpy()).all(), 'Predictores incompletos')
    require(samples.y.isin([0, 1]).all(), 'Etiqueta inválida')
    ref, target = pd.to_datetime(samples.fecha_referencia), pd.to_datetime(samples.fecha_objetivo)
    require((target-ref).dt.days.eq(samples.horizonte_dias).all(), 'Horizonte incorrecto')
    require((ref-pd.to_datetime(samples.ventana_inicio)).dt.days.eq(samples.ventana_dias).all(),
            'Inicio meteorológico incorrecto')
    require((ref-pd.to_datetime(samples.ventana_fin)).dt.days.eq(1).all(), 'Clima de fecha objetivo incluido')
    for horizon in [0, 7]:
        subset = samples[samples.horizonte_dias.eq(horizon)]
        a, b = [subset[subset.ventana_dias.eq(w)].set_index('muestra_id') for w in [7, 14]]
        require(set(a.index) == set(b.index), 'Poblaciones distintas entre ventanas')
        for c in ['y', 'resultado_t', 'fecha_referencia', 'fecha_objetivo', 'observacion_objetivo_id']:
            require(a[c].sort_index().equals(b[c].sort_index()), 'Objetivos distintos entre ventanas')


def run(input_dir, output_dir):
    input_dir, output_dir = input_dir.resolve(), output_dir.resolve()
    require(not output_dir.exists() or not any(output_dir.iterdir()),
            'Destino ocupado. Elegir un directorio nuevo para conservar las versiones anteriores.')
    require(not output_dir.is_relative_to(input_dir), 'Destino dentro del original')
    original_path = input_dir/'respaldo/monitoreo_deduplicado.csv'
    daily_path = input_dir/'respaldo/monitoreo_ml_14dias.csv'
    receipt_path = input_dir/'manifiesto_local.json'
    receipt = json.loads(receipt_path.read_text(encoding='utf-8'))
    hashes = {p.name: sha(p) for p in [original_path, daily_path]}
    for path in [original_path, daily_path]:
        expected = next(x['sha256'] for x in receipt['respaldo'] if Path(x['archivo']).name == path.name)
        require(sha(path) == expected, f'Fuente alterada: {path.name}')
    obs = observations(pd.read_csv(original_path, dtype={'latitud': str, 'longitud': str}))
    # usecols evita introducir índices auxiliares desde la versión histórica.
    daily = pd.read_csv(daily_path, dtype={'latitud': str, 'longitud': str}, usecols=['id_medicion', 'fecha_objetivo', 'fecha_dato',
                                            'dias_anteriores', 'latitud', 'longitud', *FEATURES])
    windows, failures = weather_windows(daily)
    samples, exclusions = make_samples(obs, windows, failures)
    verify_samples(samples)
    folds, summary, assignments = make_partitions(samples)
    require(hashes == {p.name: sha(p) for p in [original_path, daily_path]}, 'Fuente modificada durante ejecución')
    output_dir.mkdir(parents=True, exist_ok=True)
    obs.to_csv(output_dir/'observaciones.csv', index=False)
    samples.to_csv(output_dir/'muestras.csv', index=False)
    exclusions.to_csv(output_dir/'exclusiones.csv', index=False)
    assignments.to_csv(output_dir/'asignaciones.csv', index=False)
    summary.to_csv(output_dir/'resumen_particiones.csv', index=False)
    for h in [0, 7]:
        for w in [7, 14]:
            samples[samples.horizonte_dias.eq(h) & samples.ventana_dias.eq(w)].to_csv(
                output_dir/f'muestras_h{h}_{w}d.csv', index=False)
    write_json(output_dir/'particiones.json', folds)
    write_json(output_dir/'predictores.json', {'features': FEATURES, 'unidades': ['%']*3 + ['°C']*3 + ['mm'] + ['m/s']*3})
    outputs = {p.name: sha(p) for p in sorted(output_dir.iterdir()) if p.is_file()}
    manifest = dict(version='horizontes-v2-etapa-c', entradas=hashes,
                    manifiesto_etapa_b_sha256=sha(receipt_path), script_sha256=sha(__file__),
                    python=platform.python_version(), pandas=pd.__version__, numpy=np.__version__,
                    salidas=outputs, disponibilidad_operativa='no_acreditada',
                    meteorologia='diarios auditados Etapa B; 24 horas/día verificadas en respaldo horario',
                    criterio_ubicacion='coordenadas; IDs originales conservados',
                    evaluacion='enero-marzo 2026, previamente observada',
                    conteos={f'h{h}': int(samples[(samples.horizonte_dias == h) & (samples.ventana_dias == 7)].shape[0])
                             for h in [0, 7]})
    write_json(output_dir/'manifiesto.json', manifest)
    return samples, exclusions, summary, manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-dir', type=Path, default=ROOT/'data/raw/etapa_b_20261122')
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    _, _, summary, manifest = run(args.input_dir, args.output_dir)
    print(summary.to_string(index=False))
    print(json.dumps(manifest['conteos']))
