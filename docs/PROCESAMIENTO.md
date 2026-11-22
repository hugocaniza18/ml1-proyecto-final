# Procesamiento y reproducción

Conservamos una implementación oficial y los artefactos aprobados de la Etapa D. La reorganización no ajusta modelos ni modifica su selección. Los manifiestos originales mantienen sus hashes y rutas de origen; `provenance/migracion.json` relaciona esas rutas con las actuales. El código previo permanece en el commit `edb66b0`.

## Insumos y entorno

Usamos Python 3.10.11 y las versiones instaladas verificadas en `requirements-lock.txt`. Para crear el entorno instalamos `python -m pip install -r requirements-lock.txt`. `results/entorno.json` identifica el experimento aprobado y `provenance/entorno_consolidacion.json` identifica el entorno de la consolidación. En esta migración verificamos el entorno existente; no efectuamos una instalación limpia.

Los archivos individuales permanecen fuera de Git: `data/raw/`, `data/interim/`, `data/processed/`, `models/` y `results/predictions/`. Para reproducir el proyecto necesitamos acceso autorizado a esos insumos locales. El repositorio público conserva hashes, resultados agregados y código; no sustituye el acceso a las etiquetas cuya redistribución sigue pendiente de aclaración.

Conservamos los 650 registros originales en `data/raw/monitoreo/monitoreo_deduplicado.csv`, el registro de correspondencia de identificadores en `identificadores.csv` y 66 respuestas horarias ERA5 con sus tres manifiestos en `data/raw/era5/`. Los hashes del registro de identificadores y de los insumos están en `provenance/entradas_vigentes.json`. Las respuestas originales y el archivo de monitoreo se copian sin cambios.

## Comandos oficiales

Ejecutamos desde la raíz del repositorio:

```powershell
python -m unittest discover -s tests -v
python scripts/verificar.py
python scripts/verificar_integracion.py --save provenance/verificacion_migracion.json
```

La integración reconstruye en un directorio temporal los diarios desde las respuestas horarias, verifica las ventanas, reproduce las tablas y particiones, recarga los ocho modelos y repite únicamente su evaluación. No ejecuta entrenamientos, no descarga datos y no cambia los productos oficiales.

Para inspeccionar por separado los productos intermedios utilizamos destinos nuevos:

```powershell
python scripts/auditar_datos.py --output-dir .cache/auditoria-nueva
python scripts/procesar_datos.py --output-dir .cache/procesamiento-nuevo
```

El primer comando reconstruye las diez variables diarias y sus vínculos con las respuestas originales. El segundo utiliza los diarios aprobados en `data/interim/`, conserva los identificadores, construye h0 y h7 con emparejamiento exacto a siete días, agrega las ventanas y guarda exclusiones y particiones. La integración compara ambos pasos contra los artefactos aprobados. Las ventanas excluyen el día de referencia.

Si necesitamos reproducir el ajuste completo, utilizamos una raíz nueva, sin sobrescribir `results/` ni `models/`:

```powershell
python scripts/entrenar.py --output-dir .cache/reproduccion/results
python scripts/evaluar.py --results-dir .cache/reproduccion/results
python scripts/verificar.py --output-dir .cache/reproduccion/results
```

`entrenar.py` ejecuta GridSearchCV sobre los folds aprobados y congela la selección antes de que `evaluar.py` calcule las métricas de enero–marzo de 2026. Este período es evaluación histórica previamente observada. No lo usamos para optimizar parámetros ni seleccionar modelos. Los modelos de una reproducción se guardan junto a sus resultados, en `.cache/reproduccion/models/`.

`generar_informe.py` genera figuras y un reporte en un destino nuevo a partir de resultados guardados. `exportar_portal.py` prepara exclusivamente tablas agregadas y figuras en un destino nuevo; no modifica el portal ni exporta datos individuales. `ejecutar_notebooks.py` requiere rutas explícitas y no publica descargas. La actualización de notebooks y contenidos académicos queda para la siguiente parte de la Etapa E.

## Verificaciones y distribución

Comparamos datos y ventanas con tolerancia absoluta de 10⁻¹⁰ y predicciones y métricas con 10⁻¹², sin tolerancia relativa. Verificamos las poblaciones de 428 muestras h0 y 362 h7, las mismas observaciones entre ventanas, los folds de seis fechas semanales y la purga aprobada. Conservamos la incertidumbre sobre la fecha de disponibilidad operacional de etiquetas y meteorología.

`specs/` se utiliza exclusivamente para desarrollo SDD. `.gitattributes` la excluye de `git archive`; el exportador público utiliza una lista explícita de recursos agregados que tampoco incluye especificaciones. La documentación académica, los notebooks y la presentación se revisarán por separado y deberán permitir comprender el trabajo sin consultar `specs/`. La entrega y el portal actuales aún requieren esa revisión editorial.
