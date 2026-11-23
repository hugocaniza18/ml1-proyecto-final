# Roya asiática de la soja y meteorología

**Trabajo final de Machine Learning 1 · FIUNA · 2026**  
Hugo Rene Caniza Portillo y Ruben Dario Lopez Sanabria

[Portal del proyecto](https://ml1-proyecto-final-web.vercel.app/) · [Informe académico](reports/INFORME.md) · [Diccionario de datos](data/DICCIONARIO.md)

## Objetivo y alcance

Evaluar si los datos meteorológicos históricos permiten clasificar el estado **registrado** de roya asiática de la soja en puntos de monitoreo. Se estudian dos horizontes:

- **h0:** estado registrado en la fecha de referencia `t`.
- **h7:** estado registrado en la misma ubicación exactamente siete días después (`t+7`); incluye estados positivos que persisten.

Este es un **estudio retrospectivo**, no un sistema de alerta agronómica validado ni una predicción de nuevas infecciones.

## Datos y preparación

- **Roya:** registros de [Epagri/CIRAM Agroconnect](https://ciram.epagri.sc.gov.br/agroconnect/).
- **Clima:** ERA5 de ECMWF obtenido mediante [Open-Meteo](https://open-meteo.com/en/docs/historical-weather-api); atribución meteorológica [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
- **Origen:** 650 registros; 211 positivos, 217 negativos y 222 «Não Avaliado». El último estado se excluye de las etiquetas utilizables, **no** se considera negativo.
- **Muestras:** 428 para h0 y 362 para h7, cada una con dos representaciones meteorológicas de **7 o 14 días anteriores a `t`**. Las filas asociadas a ambas ventanas no son observaciones independientes.
- **Predictores V1:** diez resúmenes de humedad relativa, temperatura, precipitación y viento.

El conjunto de desarrollo se utiliza para exploración, ajuste y selección. Los modelos se comparan mediante **tres folds de validación temporal con purga de 14 días**, y se reserva enero–marzo de 2026 como evaluación histórica posterior. Se prioriza el **F1 de la clase positiva** para la selección. La meteorología de `t` o posterior no se utiliza como predictor.

## Modelos y resultados oficiales (V1)

Se comparan **regresión logística** y **Random Forest**, con referencias de clase mayoritaria y, para h7, persistencia del estado previamente observado. Resultados sobre el período histórico (umbral 0,5):

| Horizonte | Método | Muestras | Precisión | Recall | F1 |
|---|---|---:|---:|---:|---:|
| h0 | Random Forest · 7 días | 66 | 0,667 | 0,270 | 0,385 |
| h7 | Regresión logística · 14 días | 46 | 0,889 | 0,333 | 0,485 |
| h7 | Persistencia | 46 | 0,720 | 0,750 | 0,735 |

La referencia de persistencia detecta **18 de 24** positivos de h7, frente a **8 de 24** del modelo meteorológico V1. Por tanto, no se demuestra superioridad del modelo frente a esa regla simple en el histórico. [Comparación V1 completa](results/comparacion.csv).

## Experimento complementario (V2): favorabilidad meteorológica

Con las series horarias de ERA5 se calcularon cinco características exploratorias:

1. `horas_favorables`: horas con temperatura de 18 a 26,5 °C y humedad relativa ≥90 %.
2. `dias_6h_favorables`: días con al menos seis horas favorables acumuladas.
3. `racha_max_favorable_h`: máxima racha continua de horas favorables.
4. `horas_hr90`: cantidad de horas con humedad relativa ≥90 %.
5. `depresion_rocio_media_c`: diferencia media entre temperatura y punto de rocío estimado.

Los umbrales son **aproximaciones exploratorias de condiciones ambientales**, no mediciones de mojado foliar ni umbrales agronómicos validados para alerta. Se compararon dos grupos con los mismos horizontes, ventanas y particiones: **V2-A**, las cinco nuevas variables; **V2-B**, las diez variables de V1 más las cinco nuevas.

| Horizonte | V1 F1 | V2-A F1 | V2-B F1 |
|---|---:|---:|---:|
| h0 | 0,3846 | **0,4074** | 0,2500 |
| h7 | **0,4848** | 0,4000 | 0,2143 |

V2-A obtiene una mejora pequeña en h0, pero no en h7; la combinación de 15 variables no mejora el F1 histórico. Las diferencias no permiten afirmar significación estadística ni causalidad. **V1 se conserva como referencia oficial del portal y V2 como experimento complementario**, sin modificar retrospectivamente umbrales según el histórico.

[Comparación V1/V2](results/v2/comparacion_v1_v2.csv) · [Interpretación del experimento](results/v2/informe_comparativo.md)

## Organización

```text
ml1-proyecto-final/
├── data/
│   ├── raw/                  # Monitoreo y meteorología originales
│   ├── interim/              # Resúmenes y trazabilidad
│   └── processed/            # Datos V1 y, en v2/, muestras experimentales
├── notebooks/
│   ├── 01_auditoria.ipynb    # Calidad y construcción de muestras
│   ├── 02_exploracion.ipynb  # Análisis exploratorio de datos
│   └── 03_evaluacion.ipynb   # Entrenamiento, evaluación y V1/V2
├── scripts/
│   ├── procesar_datos.py    # Procesamiento V1/V2
│   ├── entrenar.py          # Modelado V1/V2
│   ├── evaluar.py           # Evaluación y comparación
│   └── proyecto.json        # Configuración V1
├── results/                  # Resultados V1 y resultados experimentales v2/
├── reports/                  # Informe académico
├── tests/                    # Pruebas automatizadas
└── web/                      # Portal Next.js y descargas
```

Los **tres notebooks** explican el procedimiento académico; los **tres scripts** contienen la implementación reutilizable. No se requieren los scripts auxiliares eliminados en la refactorización.

## Instalación y pruebas

Desde PowerShell con Git y Python 3.10.11:

```powershell
git clone https://github.com/hugocaniza18/ml1-proyecto-final.git
cd ml1-proyecto-final
python -m venv .venv
.\.venv\Scripts\python -m pip install -r requirements.txt
.\.venv\Scripts\python -m unittest discover -s tests -v
```

Los notebooks se abren en VS Code o Jupyter, en orden **01 → 02 → 03**, desde la raíz del repositorio. Si se requiere Jupyter Lab:

```powershell
.\.venv\Scripts\python -m pip install jupyterlab
.\.venv\Scripts\python -m jupyter lab
```

### Procesamiento y entrenamiento opcionales

Para no sobrescribir los resultados oficiales, usar destinos separados:

```powershell
.\.venv\Scripts\python scripts/procesar_datos.py --output-dir .cache/procesamiento
.\.venv\Scripts\python scripts/entrenar.py --output-dir .cache/reproduccion/results
.\.venv\Scripts\python scripts/evaluar.py --results-dir .cache/reproduccion/results
```

El entrenamiento V1 usa por defecto los datos procesados oficiales. Si se quiere entrenar con datos reconstruidos, debe configurarse explícitamente `--data-dir` y verificarse la estructura de archivos. Para V2, si están disponibles los datos y artefactos necesarios:

```powershell
.\.venv\Scripts\python scripts/procesar_datos.py --version v2 --output-dir data/processed/v2
.\.venv\Scripts\python scripts/entrenar.py --version v2
.\.venv\Scripts\python scripts/evaluar.py --comparar-v2
```

**Atención:** la ejecución V2 puede rechazar sobrescribir `results/v2/` ya existente; para repetirla se debe elegir un directorio nuevo. Los modelos y las predicciones grandes pueden estar excluidos de Git mediante `.gitignore`; los CSV de comparación publicados permiten revisar los resultados sin entrenar nuevamente.

## Portal

[Portal publicado](https://ml1-proyecto-final-web.vercel.app/). La versión principal visualiza los resultados oficiales de V1. Los artefactos experimentales V2 están documentados en el repositorio y el notebook 03.

Para compilar localmente:

```powershell
cd web
npm install
npm run build
cd ..
```

Las copias públicas de los notebooks están en `web/public/academico/`. Si se actualizan los notebooks, deben sincronizarse con esas copias antes de ejecutar las pruebas. La utilidad de descargas del portal está en `web/tools/`.

## Limitaciones y conclusión

La muestra histórica es limitada, hay observaciones repetidas por ubicación, la fecha registrada no acredita necesariamente la fecha de muestreo de campo y h7 no distingue infecciones nuevas de persistencia. ERA5 describe condiciones ambientales retrospectivas, no mojado foliar observado en las plantas. Los modelos meteorológicos presentan **recall bajo** en el histórico y, para h7, no superan la referencia simple de persistencia.

La V2 aporta una comparación reproducible de ingeniería de características, pero **no muestra mejora consistente**. El proyecto debe interpretarse como evaluación académica retrospectiva, no como producto listo para decisiones agronómicas.
