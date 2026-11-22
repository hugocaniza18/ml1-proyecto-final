# Verificación de la Etapa D

La batería completa aprobó 13 pruebas sintéticas y de contrato. Incluye ajuste del scaler solo con entrenamiento, contrato de diez predictores, mayoría y empate, persistencia dependiente de estado actual, desempates de selección independientes del rendimiento histórico, métricas con una sola clase y las siete pruebas de datos/tiempo de Etapa C.

Se ejecutó el experimento completo dos veces, en work/etapa_d/run01 y run02. Cada ejecución conservó su configuración, grillas, ocho modelos, predicciones de validación e histórico, métricas, aprendizaje, selección congelada, secuencia temporal y entorno.

Verificamos ambas ejecuciones contra los datos aprobados. Se recalcularon 44 registros de métricas a partir de 2603 predicciones por ejecución. Se comprobaron los conjuntos de IDs, las etiquetas, las referencias y el umbral. Se recargaron ocho modelos y se reprodujeron sus puntuaciones históricas, con error máximo 8,33×10⁻¹⁷. Los promedios y desviaciones CV coinciden con las predicciones y las búsquedas.

Ambas ejecuciones coinciden en selección, métricas y probabilidades con tolerancia absoluta 10⁻¹². La selección congelada conserva su hash y precede en la secuencia guardada a la evaluación histórica. La configuración se determinó con CV; la prueba histórica previamente observada no modificó los parámetros elegidos.

## Comandos de reproducción

Desde el worktree, con las dependencias fijadas y la caché local autorizada:

```powershell
python -m unittest discover -s tests -v
python scripts/verificar_etapa_c.py --output-dir data/processed/horizontes_v2_c02
python scripts/experimentos_etapa_d.py --output-dir work/etapa_d/run01
python scripts/experimentos_etapa_d.py --output-dir work/etapa_d/run02
python scripts/verificar_experimentos_d.py --output-dir work/etapa_d/run01 --repeat-dir work/etapa_d/run02
python scripts/verificar_experimentos_d.py --output-dir work/etapa_d/run02 --repeat-dir work/etapa_d/run01
python scripts/reportar_etapa_d.py --source work/etapa_d/run01 --destination reports/etapa_d
```

Los destinos deben ser nuevos. En otra repetición, elegir nombres diferentes y actualizar las rutas. El ajuste verifica los manifiestos de C, todos los productos de la traza horaria y los hashes de sus 66 respuestas y tres manifiestos de consulta. La ruta del respaldo está dentro de la copia local de experimentación. Se conserva la configuración declarativa en config/etapa_d.json.

Los ocho modelos y las predicciones individuales permanecen bajo work/, excluido de Git. El reporte público incluye métricas agregadas, grillas, figuras y evidencia. Los datos originales, el portal, los notebooks finales y la presentación se conservaron. La disponibilidad operacional sigue no acreditada.

Los dos documentos señalados de Etapa C se corrigieron con UTF-8. Se comprobaron por lectura estricta sus caracteres y símbolos, incluidos ×10⁻¹⁴ y 10⁻¹⁰.
