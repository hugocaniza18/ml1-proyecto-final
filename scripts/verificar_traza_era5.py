"""Reconstrucción offline ERA5 horario → diarios → ventanas de Etapa C."""
from pathlib import Path
from urllib.parse import urlsplit, parse_qs
import argparse
import json
import numpy as np
import pandas as pd
from procesar_horizontes import ROOT, FEATURES, AGGREGATIONS, location_id, require, sha, write_json

HOURLY = ['temperature_2m', 'relative_humidity_2m', 'precipitation',
          'wind_speed_10m', 'wind_direction_10m']


def verify(hourly_root, input_dir, processed, output_dir):
    require(not output_dir.exists(), 'Destino de trazabilidad ya existente')
    evidence_path = ROOT/'reports/etapa_b/respaldo_horario_hashes.json'
    evidence = {x['archivo']: x for x in json.loads(evidence_path.read_text(encoding='utf-8'))}
    response_records, manifest_records, all_days, provenance = [], [], [], []
    for folder in ['weather', 'weather_previos', 'weather_14dias']:
        manifest_path = hourly_root/folder/'manifest.json'
        manifest_records.append({'archivo': f'{folder}/manifest.json', 'sha256': sha(manifest_path)})
        manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
        for entry in manifest['requests']:
            relative = f"{folder}/{entry['id']:02d}.json"
            path = hourly_root/relative
            digest = sha(path)
            require(digest == evidence[relative]['sha256'], 'Respuesta distinta de Etapa B')
            if 'sha256' in entry:
                require(digest == entry['sha256'], 'Hash distinto del manifiesto de consulta')
            query = parse_qs(urlsplit(entry['url']).query)
            require(query['models'] == ['era5'] and query['timezone'] == ['America/Sao_Paulo']
                    and query['wind_speed_unit'] == ['ms'], 'Consulta incompatible')
            location = location_id(query['latitude'][0], query['longitude'][0])
            response = json.loads(path.read_text(encoding='utf-8'))
            require(response['utc_offset_seconds'] == -10800, 'Zona horaria distinta')
            units = response['hourly_units']
            require(units['temperature_2m'] == '°C' and units['relative_humidity_2m'] == '%'
                    and units['precipitation'] == 'mm' and units['wind_speed_10m'] == 'm/s'
                    and units['wind_direction_10m'] == '°', 'Unidades distintas')
            frame = pd.DataFrame(response['hourly']).set_index('time')[HOURLY]
            index = pd.to_datetime(frame.index)
            expected = pd.date_range(query['start_date'][0],
                                     pd.Timestamp(query['end_date'][0])+pd.Timedelta(hours=23), freq='h')
            require(index.equals(expected), 'Serie horaria incompleta o duplicada')
            require(np.isfinite(frame.to_numpy()).all(), 'Serie horaria no finita')
            frame.index = index
            frame['u'] = -frame.wind_speed_10m*np.sin(np.radians(frame.wind_direction_10m))
            frame['v'] = -frame.wind_speed_10m*np.cos(np.radians(frame.wind_direction_10m))
            for day, hours in frame.groupby(frame.index.strftime('%Y-%m-%d'), sort=True):
                require(len(hours) == 24, 'Día sin 24 horas')
                row = dict(ubicacion_id=location, fecha_dato=day,
                           hr_media_pct=hours.relative_humidity_2m.mean(),
                           hr_min_pct=hours.relative_humidity_2m.min(),
                           hr_max_pct=hours.relative_humidity_2m.max(),
                           temperatura_media_c=hours.temperature_2m.mean(),
                           temperatura_min_c=hours.temperature_2m.min(),
                           temperatura_max_c=hours.temperature_2m.max(),
                           precipitacion_mm=hours.precipitation.sum(),
                           viento_vel_media_10m_ms=hours.wind_speed_10m.mean(),
                           viento_u_media_10m_ms=hours.u.mean(), viento_v_media_10m_ms=hours.v.mean())
                all_days.append(row)
                provenance.append({'ubicacion_id': location, 'fecha_dato': day, 'respuesta': relative,
                                   'sha256_respuesta': digest, 'primera_hora': str(hours.index.min()),
                                   'ultima_hora': str(hours.index.max()), 'horas': 24})
            response_records.append({'archivo': relative, 'sha256': digest, 'url': entry['url'],
                                     'horas': len(frame), 'ubicacion_id': location,
                                     'latitud_celda': response['latitude'],
                                     'longitud_celda': response['longitude']})
    require(set(evidence) == {r['archivo'] for r in response_records}, 'Faltan respuestas de respaldo')
    rebuilt = pd.DataFrame(all_days)
    require(not rebuilt.duplicated(['ubicacion_id', 'fecha_dato']).any(), 'Solapamiento de respuestas')
    rebuilt = rebuilt.set_index(['ubicacion_id', 'fecha_dato'])
    source_daily = input_dir/'respaldo/monitoreo_ml_14dias.csv'
    receipt = json.loads((input_dir/'manifiesto_local.json').read_text(encoding='utf-8'))
    expected_daily = next(x['sha256'] for x in receipt['respaldo']
                          if Path(x['archivo']).name == source_daily.name)
    require(sha(source_daily) == expected_daily, 'Diario original modificado')
    daily = pd.read_csv(source_daily, dtype={'latitud': str, 'longitud': str},
                        usecols=['id_medicion', 'fecha_objetivo', 'dias_anteriores', 'fecha_dato',
                                 'latitud', 'longitud', *FEATURES])
    daily['ubicacion_id'] = [location_id(a, b) for a, b in zip(daily.latitud, daily.longitud)]
    keys = pd.MultiIndex.from_frame(daily[['ubicacion_id', 'fecha_dato']])
    reconstructed_values = rebuilt.loc[keys, FEATURES].to_numpy()
    errors_daily = np.max(np.abs(reconstructed_values-daily[FEATURES].to_numpy()), axis=0)
    require(np.all(errors_daily <= 1e-10), 'Diarios distintos del horario')
    reconstructed_daily = daily.copy()
    reconstructed_daily[FEATURES] = reconstructed_values
    samples_path = processed/'muestras.csv'
    samples = pd.read_csv(samples_path)
    errors_windows = {}
    for window in [7, 14]:
        aggregation = reconstructed_daily[reconstructed_daily.dias_anteriores.le(window)].groupby(
            'id_medicion').agg(AGGREGATIONS)
        subset = samples[samples.ventana_dias.eq(window)]
        values = aggregation.loc[subset.id_medicion_anterior, FEATURES].to_numpy()
        errors = np.max(np.abs(values-subset[FEATURES].to_numpy()), axis=0)
        require(np.all(errors <= 1e-10), 'Ventanas distintas del horario original')
        errors_windows[str(window)] = dict(zip(FEATURES, errors.tolist()))
    # Traza de cada fila del diario recibido a la respuesta horaria.
    links = daily[['id_medicion', 'fecha_objetivo', 'dias_anteriores', 'ubicacion_id', 'fecha_dato']].merge(
        pd.DataFrame(provenance), on=['ubicacion_id', 'fecha_dato'], validate='many_to_one')
    require(len(links) == len(daily), 'Traza diaria incompleta')
    output_dir.mkdir(parents=True)
    links.to_csv(output_dir/'traza_diarios_respuestas.csv', index=False)
    reconstructed_daily.to_csv(output_dir/'diarios_reconstruidos.csv', index=False)
    summary = dict(respuestas_verificadas=len(response_records),
                   horas_respaldadas=sum(x['horas'] for x in response_records),
                   dias_unicos_respaldados=len(rebuilt), filas_diarias_verificadas=len(daily),
                   dias_requeridos_unicos=len(set(keys)), filas_ventanas_verificadas=len(samples),
                   error_maximo_diarios=float(errors_daily.max()),
                   error_maximo_ventanas=max(max(e.values()) for e in errors_windows.values()),
                   tolerancia_absoluta=1e-10, descargas=0,
                   evidencia_b_sha256=sha(evidence_path), diario_fuente_sha256=sha(source_daily),
                   muestras_sha256=sha(samples_path), script_sha256=sha(__file__),
                   disponibilidad_operativa='no_acreditada')
    write_json(output_dir/'traza.json', {'resumen': summary, 'manifiestos': manifest_records,
                                       'respuestas': response_records,
                                       'errores_diarios': dict(zip(FEATURES, errors_daily.tolist())),
                                       'errores_ventanas': errors_windows,
                                       'salidas': {p.name: sha(p) for p in sorted(output_dir.iterdir())}})
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--hourly-root', type=Path, required=True)
    parser.add_argument('--input-dir', type=Path, default=ROOT/'data/raw/etapa_b_20261122')
    parser.add_argument('--processed-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(verify(args.hourly_root, args.input_dir, args.processed_dir, args.output_dir),
                     ensure_ascii=False, indent=2))
