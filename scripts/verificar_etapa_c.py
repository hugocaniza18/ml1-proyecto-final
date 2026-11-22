"""Verifica salidas de C contra insumos auditados y ejecución independiente."""
from pathlib import Path
import argparse
import json
import tempfile
import pandas as pd
import numpy as np
from procesar_horizontes import ROOT, FEATURES, run, sha, verify_samples, require


def verify(input_dir, output_dir):
    samples = pd.read_csv(output_dir/'muestras.csv')
    verify_samples(samples)
    original = input_dir/'respaldo/monitoreo_deduplicado.csv'
    before = sha(original)
    received = pd.read_csv(input_dir/'royasoja_dataset_sin_iith/datos_horizontes_0_y_7.csv').set_index('id_medicion')
    for h, n in [(0, 428), (7, 362)]:
        for w in [7, 14]:
            subset = samples[samples.horizonte_dias.eq(h) & samples.ventana_dias.eq(w)]
            require(len(subset) == n, 'Población distinta de la aprobada')
            reference = received.loc[subset.id_medicion_anterior]
            target = 'objetivo_dia_0' if h == 0 else 'objetivo_dia_mas_7'
            require(np.array_equal(subset.y, reference[target]), 'Etiquetas diferentes al antecedente')
            for col in FEATURES:
                require(np.allclose(subset[col], reference[f'{col}_{w}d'], atol=1e-10, rtol=0),
                        'Diferencia de agregación meteorológica')
    partitions = json.loads((output_dir/'particiones.json').read_text(encoding='utf-8'))
    expected = {0: [(164, 38), (188, 55), (239, 100)],
                7: [(141, 31), (150, 51), (196, 90)]}
    for horizon in [0, 7]:
        population = samples[(samples.horizonte_dias == horizon) & (samples.ventana_dias == 14)].set_index('muestra_id')
        for number, (ntrain, nval) in enumerate(expected[horizon], 1):
            block = partitions[str(horizon)][str(number)]
            require((len(block['train_ids']), len(block['validation_ids'])) == (ntrain, nval),
                    'Tamaños distintos de folds aprobados')
            valid = population.loc[block['validation_ids']]
            dates = pd.to_datetime(valid.fecha_objetivo).drop_duplicates().sort_values()
            require(len(dates) == 6 and dates.diff().dropna().dt.days.eq(7).all(), 'Bloque semanal discontinuo')
        historical = partitions[str(horizon)]['historico']
        require((len(historical['train_ids']), len(historical['validation_ids']), len(historical['purged_ids']))
                == ((359, 66, 3) if horizon == 0 else (313, 46, 3)), 'Corte histórico distinto')
    excluded = pd.read_csv(output_dir/'exclusiones.csv')
    require(excluded.motivo.eq('referencia_no_evaluado').sum() == 444, 'Referencias no evaluadas mal contadas')
    require(excluded.motivo.eq('objetivo_no_evaluado').sum() == 56, 'Exclusiones h7 no evaluadas distintas')
    require(excluded.motivo.eq('objetivo_sin_registro').sum() == 10, 'Exclusiones h7 sin registro distintas')
    require(set(excluded.motivo) == {'referencia_no_evaluado', 'objetivo_no_evaluado', 'objetivo_sin_registro'},
            'Exclusión no prevista en insumos actuales')
    manifest = json.loads((output_dir/'manifiesto.json').read_text(encoding='utf-8'))
    for name, digest in manifest['salidas'].items():
        require(sha(output_dir/name) == digest, 'Salida alterada')
    with tempfile.TemporaryDirectory(prefix='etapa_c_') as temp:
        other = Path(temp)/'repeticion'
        _, _, _, second = run(input_dir, other)
        require(manifest == second, 'Ejecuciones no idénticas')
        try:
            run(input_dir, other)
        except ValueError:
            pass
        else:
            raise ValueError('Se permitió sobrescribir una versión')
    require(sha(original) == before, 'Original modificado')
    return {'poblaciones': {'h0': 428, 'h7': 362}, 'filas_muestras': len(samples),
            'variables': len(FEATURES), 'folds_verificados': 6, 'reproduccion_hashes': True,
            'destino_ocupado_rechazado': True, 'antecedentes_coincidentes': True,
            'disponibilidad_operativa': 'no_acreditada', 'entrenamientos': 0}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-dir', type=Path, default=ROOT/'data/raw/etapa_b_20261122')
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(verify(args.input_dir, args.output_dir), ensure_ascii=False, indent=2))
