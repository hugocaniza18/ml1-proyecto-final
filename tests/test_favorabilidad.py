"""Pruebas del cálculo de variables de favorabilidad V2."""
import sys
import unittest
from pathlib import Path
import pandas as pd
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from procesar_datos import favorable_summary, make_favorability_dataset


class FavorabilityTests(unittest.TestCase):
    def test_empty_and_incomplete_window_rejected(self):
        with self.assertRaises(ValueError):
            favorable_summary([20] * 23, [95] * 23)

    def test_favorable_thresholds_and_dewpoint(self):
        result = favorable_summary([20] * 24, [95] * 24)
        self.assertEqual(result['horas_favorables'], 24)
        self.assertEqual(result['dias_6h_favorables'], 1)
        self.assertEqual(result['racha_max_favorable_h'], 24)
        self.assertEqual(result['horas_hr90'], 24)
        self.assertGreater(result['depresion_rocio_media_c'], 0)

    def test_streak_across_midnight(self):
        temperature = [10] * 48
        humidity = [50] * 48
        for hour in range(20, 28):
            temperature[hour] = 20
            humidity[hour] = 95
        result = favorable_summary(temperature, humidity)
        self.assertEqual(result['racha_max_favorable_h'], 8)
        self.assertEqual(result['dias_6h_favorables'], 0)  # 4 horas por día

    def test_bad_humidity_rejected(self):
        with self.assertRaises(ValueError):
            favorable_summary([20] * 24, [101] * 24)

    def test_future_weather_rejected(self):
        fake = pd.DataFrame([{
            'muestra_id': 'm1', 'ventana_dias': 7, 'horizonte_dias': 7,
            'ubicacion_id': '0,0', 'ventana_inicio': '2025-01-01',
            'ventana_fin': '2025-01-07', 'fecha_referencia': '2025-01-07',
        }])
        with patch('procesar_datos.load_hourly_era5', return_value={'0,0': {}}):
            with self.assertRaisesRegex(ValueError, 'referencia'):
                make_favorability_dataset(fake, Path('/not/a/dataset'))


if __name__ == '__main__':
    unittest.main()
