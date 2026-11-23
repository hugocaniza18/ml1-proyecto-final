# Roya de la soja y meteorología

Trabajo final de Machine Learning 1 · FIUNA · 2026

Hugo Rene Caniza Portillo y Ruben Dario Lopez Sanabria

[Portal en Vercel](https://ml1-proyecto-final-b2uvlo4xv-hugo1-75fb.vercel.app/) · [Informe académico](reports/INFORME.md)

Estudiamos cuánto aporta la meteorología para clasificar el estado de roya asiática registrado en un punto de monitoreo. Evaluamos dos horizontes: **h0**, el estado registrado en la fecha de referencia, y **h7**, el estado registrado en la misma ubicación exactamente siete días después, incluidos los positivos que continúan de una semana a otra.

## Datos y método

El repositorio incluye los datos originales de monitoreo y las respuestas horarias de ERA5 en `data/raw/`, los resúmenes diarios y su trazabilidad en `data/interim/`, y las muestras, exclusiones y particiones en `data/processed/`. Partimos de 650 registros. Obtuvimos 428 muestras para h0 y 362 para h7. [Diccionario de datos](data/DICCIONARIO.md)

Para cada muestra resumimos diez variables de humedad, temperatura, precipitación y viento durante los **7 o 14 días anteriores** a la referencia. Comparamos regresión logística y Random Forest en ambas ventanas. Seleccionamos por F1 de la clase positiva en tres bloques de validación temporal con una purga de 14 días. Usamos la clase mayoritaria como referencia en ambos horizontes y la persistencia en h7. [Particiones](results/academicos/particiones.csv)

Las etiquetas provienen de [Epagri/CIRAM Agroconnect](https://ciram.epagri.sc.gov.br/agroconnect/). La meteorología corresponde al reanálisis ERA5 de ECMWF, obtenido mediante [Open-Meteo](https://open-meteo.com/en/docs/historical-weather-api) y agregado por día y ventana, con atribución [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) para los datos meteorológicos.

## Resultados

Evaluación histórica de enero a marzo de 2026, período previamente observado. Las métricas corresponden a la clase positiva. [Comparación completa](results/comparacion.csv)

| Horizonte | Método | Muestras | Precisión | Sensibilidad | F1 |
|---|---|---:|---:|---:|---:|
| h0 | Random Forest · 7 días | 66 | 0,667 | 0,270 | 0,385 |
| h7 | Regresión logística · 14 días | 46 | 0,889 | 0,333 | 0,485 |
| h7 | Persistencia | 46 | 0,720 | 0,750 | 0,735 |

En h7, repetir el estado observado en la fecha de referencia recuperó 18 de los 24 positivos de la semana siguiente. El modelo meteorológico recuperó 8. **Persistencia obtuvo el mayor F1** en esas 46 muestras. Esta comparación retrospectiva utiliza información distinta para cada método y resulta insuficiente para validar una alerta operativa.

## Recorrido del repositorio

| Notebook | Qué muestra |
|---|---|
| [01 · Auditoría](notebooks/01_auditoria.ipynb) | Etiquetas, exclusiones, construcción de h0/h7 y separación temporal. |
| [02 · Exploración](notebooks/02_exploracion.ipynb) | Distribuciones meteorológicas, correlaciones y cobertura del conjunto de desarrollo. |
| [03 · Evaluación](notebooks/03_evaluacion.ipynb) | Comparación de modelos y referencias, matrices de confusión y curvas a partir de resultados guardados. |

`scripts/` contiene el procesamiento, el entrenamiento y la evaluación. `tests/` reúne las pruebas, `results/` conserva los resultados oficiales, `reports/` contiene el informe y sus figuras, y `web/` aloja el portal estático.

## Reproducción

En PowerShell, con Git y Python 3.10.11, clonamos la rama del proyecto e instalamos las dependencias en un entorno virtual:

```powershell
git clone --branch feat/reformulacion-horizontes https://github.com/hugocaniza18/ml1-proyecto-final.git
cd ml1-proyecto-final
python -m venv .venv
.\.venv\Scripts\python -m pip install -r requirements.txt
```

Desde la raíz, estos comandos ejecutan las pruebas, reconstruyen las muestras en una carpeta nueva y comprueban los tres notebooks conservando sus salidas guardadas:

```powershell
.\.venv\Scripts\python -m unittest discover -s tests -v
.\.venv\Scripts\python scripts/procesar_datos.py --output-dir .cache/procesamiento
.\.venv\Scripts\python scripts/ejecutar_notebooks.py --comprobar (Get-ChildItem notebooks/*.ipynb).FullName
```

Para repetir el ajuste y la evaluación, usamos un destino nuevo. El primer comando selecciona los modelos mediante validación y guarda la selección. El segundo evalúa el período histórico:

```powershell
.\.venv\Scripts\python scripts/entrenar.py --output-dir .cache/reproduccion/results
.\.venv\Scripts\python scripts/evaluar.py --results-dir .cache/reproduccion/results
```
