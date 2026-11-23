# Predicción de roya de soja con datos climáticos

Trabajo final de Machine Learning 1, Facultad de Ingeniería, Universidad Nacional de Asunción, 2026.

**Integrantes:** [Hugo Rene Caniza Portillo](https://github.com/hugocaniza18) y [Ruben Dario Lopez Sanabria](https://github.com/Rub3nL0pez).

La implementación vigente utiliza diez variables meteorológicas, ventanas de 7 y 14 días y horizontes h0 y h7. Conservamos los ocho experimentos aprobados de regresión logística y Random Forest y las referencias de mayoría y persistencia. Enero–marzo de 2026 es evaluación histórica previamente observada.

## Material académico

El [informe](reports/INFORME.md) presenta el experimento y sus limitaciones. En [notebooks/](notebooks/) conservamos la auditoría, la exploración del conjunto de desarrollo y la evaluación con sus gráficos y tablas. Los resultados agregados, la selección congelada, los hiperparámetros y el entorno se encuentran en [results/](results/). Las fechas, tamaños y clases por bloque se pueden consultar en [particiones.csv](results/academicos/particiones.csv).

## Datos y acceso

Usamos etiquetas de [Epagri/CIRAM Agroconnect](https://ciram.epagri.sc.gov.br/agroconnect/) y meteorología ERA5 obtenida mediante [Open-Meteo](https://open-meteo.com/) con atribución [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Agregamos las series horarias por día y por ventana. El [diccionario](data/DICCIONARIO.md) describe las muestras.

La redistribución de los registros individuales de monitoreo está pendiente de autorización. Para repetir el procesamiento se necesita acceso autorizado al monitoreo y a los insumos locales de la misma versión. Las rutas esperadas son:

- `data/raw/monitoreo/` para los 650 registros en `monitoreo_deduplicado.csv` y su correspondencia en `identificadores.csv`.
- `data/raw/era5/` para los tres lotes horarios con sus manifiestos.
- `data/interim/` para los diarios aprobados y su traza. En `data/processed/` conservamos las tablas por horizonte y ventana, exclusiones y asignaciones por muestra.
- `models/` y `results/predictions/` para verificar localmente las predicciones individuales con los modelos congelados.

Las huellas de entradas, salidas procesadas y artefactos del experimento están en [data/huellas.json](data/huellas.json). El repositorio público permite consultar los tres notebooks y ejecutar el de evaluación con sus agregados. La auditoría y el EDA requieren los insumos autorizados.

## Reproducción

Usamos Python 3.10.11. Desde la raíz, en PowerShell:

```powershell
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python scripts/ejecutar_notebooks.py --comprobar notebooks/03_evaluacion.ipynb
```

Con los insumos locales podemos ejecutar los tres notebooks y verificar la integración. La verificación reconstruye diarios, ventanas y particiones en una carpeta temporal, y reevalúa los modelos guardados. Las tolerancias absolutas son 1e-10 para el procesamiento y 1e-12 para métricas y predicciones.

```powershell
python scripts/ejecutar_notebooks.py --comprobar (Get-ChildItem notebooks/*.ipynb).FullName
python scripts/verificar_integracion.py
```

Para reproducir por separado cada etapa usamos destinos nuevos. Los parámetros se encuentran en [scripts/proyecto.json](scripts/proyecto.json). El entrenamiento selecciona mediante validación temporal y guarda la selección antes de la evaluación histórica.

```powershell
python scripts/auditar_datos.py --output-dir .cache/auditoria
python scripts/procesar_datos.py --output-dir .cache/procesamiento
python scripts/entrenar.py --output-dir .cache/reproduccion/results
python scripts/evaluar.py --results-dir .cache/reproduccion/results
```

## Portal

El portal usa datos agregados y ofrece el informe y los tres notebooks. Para sincronizar las descargas ejecutamos `python scripts/actualizar_descargas_informe.py`. Vercel usa `web/` como raíz. Para probar el sitio estático localmente:

```powershell
cd web
npm ci
npm run build
python -m http.server 8000 --directory out
```
