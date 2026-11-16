# Predicción de roya de soja con datos climáticos

TP de Machine Learning 1 (23433), FIUNA, Ingeniería Electrónica, 2026.

**Equipo:** Hugo Rene Caniza Portillo (`hcaniza@fiuna.edu.py`, [GitHub](https://github.com/hugocaniza18)) y Ruben Dario Lopez Sanabria (`rubenlopez@fiuna.edu.py`, [GitHub](https://github.com/Rub3nL0pez)).

Clasificación retrospectiva del resultado publicado por CIRAM a partir de clima ERA5 de los 7 o 14 días anteriores. El modelo devuelve una puntuación de positivo y clase a umbral 0,5. No es una alerta en tiempo real.

## Resultado principal

428 objetivos binarios, 22 coordenadas, cobertura completa de 14 días. 5.992 filas largas, **428 muestras de aprendizaje**. Desarrollo: 359; purga final: 3; prueba: 66. La validación seleccionó Random Forest de 7 días sin IITH: F1 CV 0,653 y F1 prueba 0,473. Recall prueba 0,351. Los 14 días no mejoraron de forma consistente. No se reajustó después de observar prueba.

## Consultar el portal

Con Node.js 24, desde la raíz del repositorio:

```powershell
cd web
npm ci
npm run dev
```

Abre la URL que indique Next.js. Para generar la versión estática ejecuta `npm run build`; los archivos quedan en `web/out`. El portal usa datos precalculados y funciona con esta edición pública. [Repositorio en GitHub](https://github.com/hugocaniza18/ml1-proyecto-final).

## Estructura

```text
data/raw/          originales y respuestas horarias Open-Meteo
data/processed/    objetivos, ventanas largas, features y particiones
scripts/           procesamiento, experimentos, controles y exportación
notebooks/         auditoría, EDA y revisión experimental
models/            ocho pipelines con lista de predictores
reports/           métricas, predicciones, figuras e informe
docs/              metodología, diccionario, fuentes y despliegue
presentation/      presentación y guion
web/               Next.js, TypeScript y Tailwind; exportación estática
```

## Reproducción local

Ejecuta la siguiente secuencia **desde el paquete local completo** con Python 3.10. Incluye los CSV de entrada y las respuestas horarias guardadas. La estructura de datos y modelos mostrada arriba corresponde a ese paquete.

```powershell
python -m venv .venv
.venv/Scripts/python.exe -m pip install -r requirements-lock.txt
.venv/Scripts/python.exe scripts/data_pipeline.py
.venv/Scripts/python.exe scripts/experiments.py --evaluate
.venv/Scripts/python.exe scripts/verify.py
.venv/Scripts/python.exe scripts/build_reports.py
.venv/Scripts/python.exe scripts/execute_notebooks.py
```

Si faltan respuestas horarias se puede habilitar su descarga con `scripts/data_pipeline.py --download`. Solo envía coordenadas, fechas y variables meteorológicas a Open-Meteo. La ejecución por defecto usa la caché local, sin red.

La selección se escribe antes de acceder a prueba en `reports/seleccion_congelada.json`. Reejecutar exige que coincidan datos, particiones y selección; no sobrescribe silenciosamente un experimento distinto. Una nueva investigación debe usar otra versión y una nueva prueba, no optimizar contra enero–marzo de 2026.

Las ocho variantes son LR/RF × 7/14 días × con/sin IITH. No se utiliza DummyClassifier. `scripts/verify.py` comprueba correspondencia de ventanas, purgas, métricas y reproducción de probabilidades con modelos guardados.

## Lecturas y entrega

- [Informe completo](reports/INFORME.md)
- [Presentación PowerPoint](presentation/Presentacion_TP_Roya.pptx)
- [Metodología y entrada/salida](docs/METODOLOGIA.md)
- [Diccionario](docs/DICCIONARIO.md)
- [Fuentes y redistribución](docs/FUENTES_Y_LICENCIAS.md)
- [Publicación en Vercel](docs/PUBLICAR.md)
- [Referencias de la cátedra](https://github.com/carlobeni/Machine-Learning)

Los CSV originales no se alteran. HMF y meteorología de estaciones CIRAM siguen pendientes. La licencia de redistribución de las etiquetas debe aclararse antes de publicar los datos. La copia local contiene los artefactos completos.

Meteorología: [Weather data by Open-Meteo.com](https://open-meteo.com/), ERA5, [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Se realizaron agregaciones y cálculos de IITH y viento. Etiquetas: [Epagri/CIRAM Agroconnect](https://ciram.epagri.sc.gov.br/agroconnect/).

## Edición pública

Incluye métricas, figuras y seis ejemplos sin identificadores de origen. El paquete local completo contiene también las respuestas horarias y los CSV necesarios para ejecutar los scripts. Para reproducir el TP, usa el paquete local. Las instrucciones de publicación de esta edición están en docs/PUBLICAR.md.
