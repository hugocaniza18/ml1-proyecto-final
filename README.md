# Roya de la soja y meteorología

**Trabajo final de Machine Learning 1 · FIUNA · 2026**  
Hugo Rene Caniza Portillo y Ruben Dario Lopez Sanabria

[Portal del proyecto](https://ml1-proyecto-final-web.vercel.app/) · [Informe académico](reports/INFORME.md) · [Diccionario de datos](data/DICCIONARIO.md)

## Objetivo

Analizar cuánto aporta la información meteorológica a la clasificación del estado de **roya asiática de la soja** registrado en los puntos de monitoreo. Se consideran dos horizontes:

- **h0:** estado registrado en la fecha de referencia.
- **h7:** estado registrado en la misma ubicación exactamente siete días después, incluyendo positivos que persisten entre visitas.

Es un estudio retrospectivo y no representa un sistema de alerta operativa validado.

## Datos y metodología

Se utilizan observaciones de [Epagri/CIRAM Agroconnect](https://ciram.epagri.sc.gov.br/agroconnect/) y meteorología ERA5 de ECMWF, obtenida mediante [Open-Meteo](https://open-meteo.com/en/docs/historical-weather-api). Los datos meteorológicos se atribuyen bajo [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

A partir de **650 registros originales**, el procesamiento produjo **428 muestras para h0** y **362 para h7**. Se construyeron diez predictores meteorológicos de humedad, temperatura, precipitación y viento, calculados sobre los **7 o 14 días anteriores** a la fecha de referencia.

Se compararon **regresión logística** y **Random Forest**, empleando tres bloques de validación temporal con **purga de 14 días**. La selección se realizó según el **F1 de la clase positiva**. Como referencias se utilizaron el clasificador mayoritario y, para h7, la persistencia del estado observado.

Las definiciones de variables y exclusiones se documentan en el [diccionario de datos](data/DICCIONARIO.md), y las particiones resumidas en [results/academicos/particiones.csv](results/academicos/particiones.csv).

## Resultados principales

Evaluación histórica de enero a marzo de 2026, sobre un período previamente observado. Las métricas de precisión, sensibilidad y F1 se refieren a la clase positiva.

| Horizonte | Método | Muestras | Precisión | Sensibilidad (recall) | F1 |
|---|---|---:|---:|---:|---:|
| h0 | Random Forest · 7 días | 66 | 0,667 | 0,270 | 0,385 |
| h7 | Regresión logística · 14 días | 46 | 0,889 | 0,333 | 0,485 |
| h7 | Persistencia | 46 | 0,720 | 0,750 | 0,735 |

Para h7, la persistencia identificó **18 de los 24 positivos** de la semana siguiente, mientras que el modelo meteorológico identificó **8**. En estas muestras, **persistencia obtuvo un F1 superior**. Esto limita la interpretación predictiva de los modelos basados solamente en las variables meteorológicas evaluadas.

[Consultar la comparación completa](results/comparacion.csv)

## Organización del proyecto

```text
ml1-proyecto-final/
├── data/
│   ├── raw/                 # Observaciones y respuestas meteorológicas originales
│   ├── interim/             # Resúmenes diarios y trazabilidad
│   └── processed/           # Muestras y particiones listas para ML
├── notebooks/
│   ├── 01_auditoria.ipynb   # Calidad, etiquetas, exclusiones y particiones
│   ├── 02_exploracion.ipynb # Análisis exploratorio y gráficos
│   └── 03_evaluacion.ipynb  # Entrenamiento, predicciones y evaluación
├── scripts/
│   ├── procesar_datos.py    # Construcción de datos y particiones
│   ├── entrenar.py          # Entrenamiento y selección de modelos
│   ├── evaluar.py           # Evaluación y exportaciones
│   └── proyecto.json        # Parámetros y rutas del proyecto
├── results/                 # Métricas y resultados oficiales
├── reports/                 # Informe y figuras
├── tests/                   # Pruebas automatizadas
└── web/                     # Portal web y recursos publicados
```

Los notebooks están pensados para estudiar y explicar el procedimiento de ML1; los scripts concentran la lógica reutilizable. El portal usa recursos de `web/` y copias de los notebooks en `web/public/academico/`.

## Instalación

Desde **PowerShell**, con Git y Python **3.10.11**:

```powershell
git clone https://github.com/hugocaniza18/ml1-proyecto-final.git
cd ml1-proyecto-final
python -m venv .venv
.\.venv\Scripts\python -m pip install -r requirements.txt
```

> Si ya se clonó el repositorio, no es necesario volver a clonarlo. Ejecutar los comandos siguientes desde su carpeta raíz y utilizando el entorno instalado.

## Notebooks

Abrir Jupyter desde la raíz del proyecto y ejecutar en este orden:

1. `notebooks/01_auditoria.ipynb`: revisión de calidad, construcción de etiquetas y controles temporales.
2. `notebooks/02_exploracion.ipynb`: distribuciones, diferencias por clase, correlaciones y cobertura.
3. `notebooks/03_evaluacion.ipynb`: modelos, entrenamiento, predicciones y métricas.

Se pueden abrir con VS Code o Jupyter. Para usar Jupyter Lab, instalarlo en el entorno si todavía no está disponible:

```powershell
.\.venv\Scripts\python -m pip install jupyterlab
.\.venv\Scripts\python -m jupyter lab
```

## Pruebas y reproducción

Ejecutar las pruebas automatizadas:

```powershell
.\.venv\Scripts\python -m unittest discover -s tests -v
```

Para volver a generar datos, entrenamiento y evaluación, utilizar **carpetas nuevas**: esto evita sobrescribir los resultados oficiales del repositorio.

```powershell
.\.venv\Scripts\python scripts/procesar_datos.py --output-dir .cache/procesamiento
.\.venv\Scripts\python scripts/entrenar.py --output-dir .cache/reproduccion/results
.\.venv\Scripts\python scripts/evaluar.py --results-dir .cache/reproduccion/results
```

**Nota:** las muestras de `data/processed/` y los archivos de `results/` son la referencia oficial vigente. Los comandos anteriores permiten repetir etapas, pero no implican automáticamente que el entrenamiento consuma la carpeta nueva de procesamiento: para eso hay que indicar `--data-dir` y verificar que contenga los archivos esperados. Los modelos y predicciones regenerados pueden depender de archivos no versionados.

## Portal web

El portal está desarrollado con Next.js y utiliza resultados exportados a archivos estáticos. Su implementación está separada de los scripts principales de ML.

Para comprobar que compila:

```powershell
cd web
npm install
npm run build
cd ..
```

Para iniciar el servidor local:

```powershell
cd web
npm run dev
```

La utilidad `web/tools/actualizar_descargas_informe.py` sincroniza recursos académicos publicados. Los notebooks de `notebooks/` deben mantenerse iguales a sus copias en `web/public/academico/` cuando se actualicen las descargas del portal.

Las funciones de exportación académica y del resumen del portal están integradas en `scripts/evaluar.py`; no es necesario volver a incorporar los scripts auxiliares eliminados. **No ejecutar exportaciones sobre archivos oficiales sin revisar previamente el destino**, ya que pueden sobrescribir recursos utilizados por el portal.

## Alcance y próximos pasos

La implementación actual conserva la comparación oficial basada en diez predictores y ventanas meteorológicas de 7/14 días. Una etapa posterior evaluará variables de **favorabilidad meteorológica** derivadas de las series horarias, manteniendo la separación temporal y una comparación explícita con los resultados originales.
