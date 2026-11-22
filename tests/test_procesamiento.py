"""Regresiones concretas del contrato temporal y de etiquetas. Sin entrenamientos."""
from pathlib import Path
import sys
import unittest
from io import StringIO

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from procesar_horizontes import (FEATURES, observations, weather_windows, make_samples,
                                verify_samples, check_separation, run, location_id)


def original(dates, labels):
    return pd.DataFrame([
        dict(latitud=-27, longitud=-52, fecha_registro=date, punto_id=1,
             clase_original=label, clase_codigo={'Negativo': 80, 'Positivo': 79, 'Não Avaliado': 82}[label],
             conflicto_clase_mismo_punto_fecha=0, fuente_url='test')
        for date, label in zip(dates, labels)])


def daily(date='2025-12-29'):
    rows = []
    for lag in range(1, 15):
        row = dict(id_medicion='M1', fecha_objetivo=date,
                   fecha_dato=str((pd.Timestamp(date)-pd.Timedelta(days=lag)).date()),
                   dias_anteriores=lag, latitud=-27, longitud=-52)
        row.update(dict(zip(FEATURES, [80, 60, 100, 20, 10, 30, 2, 3, 1, 1])))
        rows.append(row)
    return pd.DataFrame(rows)


class ContractTests(unittest.TestCase):
    def test_coordinate_identity_preserves_distinct_source_decimals(self):
        self.assertNotEqual(location_id('-27.12345678901', '-52'),
                            location_id('-27.12345678902', '-52'))
        self.assertNotEqual(location_id('-27.123456789012345678901', '-52'),
                            location_id('-27.123456789012345678902', '-52'))
        self.assertEqual(location_id('-2.70e1', '-52.000'), location_id('-27', '-52'))
        self.assertEqual(location_id('-0.0', '0.000'), location_id('0', '0'))
        raw = original(['2025-12-29', '2025-12-29'], ['Negativo', 'Positivo'])
        raw['latitud'] = ['-27.123456789012345678901', '-27.123456789012345678902']
        restored = pd.read_csv(StringIO(raw.to_csv(index=False)),
                               dtype={'latitud': str, 'longitud': str})
        observed = observations(restored)
        self.assertEqual(observed.ubicacion_id.nunique(), 2)
        self.assertEqual(observed.observacion_id.nunique(), 2)
        self.assertTrue(observed.estado.eq('evaluado').all())
        for value in ['NaN', 'Infinity']:
            with self.assertRaises(ValueError):
                location_id(value, '0')

    def test_exact_pair_crosses_year_and_windows_stay_before_reference(self):
        obs = observations(original(['2025-12-29', '2026-01-05'], ['Negativo', 'Positivo']))
        weather, failures = weather_windows(daily())
        samples, _ = make_samples(obs, weather, failures)
        future = samples[samples.horizonte_dias.eq(7)]
        self.assertEqual(len(future), 2)
        self.assertTrue(future.y.eq(1).all())
        self.assertTrue(future.resultado_t.eq(0).all())
        self.assertTrue(future.ventana_fin.eq('2025-12-28').all())
        self.assertEqual(future.iloc[0].precipitacion_mm, 14)
        self.assertEqual(future.iloc[1].precipitacion_mm, 28)
        verify_samples(samples)

    def test_later_visit_does_not_replace_missing_day(self):
        obs = observations(original(['2025-12-29', '2026-01-12'], ['Negativo', 'Positivo']))
        weather, failures = weather_windows(daily())
        samples, excluded = make_samples(obs, weather, failures)
        self.assertFalse(samples.horizonte_dias.eq(7).any())
        self.assertIn('objetivo_sin_registro', set(excluded.motivo))

    def test_unevaluated_and_conflict_are_not_negative(self):
        obs = observations(original(['2025-12-29', '2026-01-05'], ['Negativo', 'Não Avaliado']))
        weather, failures = weather_windows(daily())
        samples, excluded = make_samples(obs, weather, failures)
        self.assertFalse(samples.horizonte_dias.eq(7).any())
        self.assertIn('objetivo_no_evaluado', set(excluded.motivo))
        raw = original(['2025-12-29', '2026-01-05', '2026-01-05'],
                       ['Negativo', 'Positivo', 'Negativo'])
        obs = observations(raw)
        self.assertEqual(obs.estado.eq('conflicto').sum(), 2)
        _, excluded = make_samples(obs, weather, failures)
        self.assertIn('objetivo_conflicto', set(excluded.motivo))

    def test_missing_or_nonfinite_weather_excludes_both_windows(self):
        for frame in [daily().iloc[:-1].copy(), daily()]:
            if len(frame) == 14:
                frame.loc[0, 'hr_media_pct'] = float('nan')
            weather, failures = weather_windows(frame)
            self.assertFalse(weather)
            self.assertEqual(len(failures), 1)

    def test_duplicates_and_target_day_weather_fail(self):
        frame = daily()
        with self.assertRaises(ValueError):
            weather_windows(pd.concat([frame, frame.iloc[:1]]))
        frame.loc[0, 'fecha_dato'] = frame.loc[0, 'fecha_objetivo']
        with self.assertRaises(ValueError):
            weather_windows(frame)

    def test_future_label_at_purge_boundary_is_rejected(self):
        def block(prefix, reference, target):
            return pd.DataFrame(dict(muestra_id=[prefix+'0', prefix+'1'], y=[0, 1],
                                     fecha_referencia=[reference]*2, fecha_objetivo=[target]*2,
                                     ventana_inicio=['2025-08-04']*2,
                                     ventana_fin=['2025-07-20']*2))
        train = block('a', '2025-07-28', '2025-08-04')
        valid = block('b', '2025-08-18', '2025-08-25')
        with self.assertRaises(ValueError):
            check_separation(train, valid)


if __name__ == '__main__':
    unittest.main()
