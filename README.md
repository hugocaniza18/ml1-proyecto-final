# Predicción de roya de soja con datos climáticos

Trabajo final de Machine Learning 1, Facultad de Ingeniería, Universidad Nacional de Asunción, 2026.

**Integrantes:** [Hugo Rene Caniza Portillo](https://github.com/hugocaniza18) y [Ruben Dario Lopez Sanabria](https://github.com/Rub3nL0pez).

La implementación vigente utiliza diez variables meteorológicas, ventanas de 7 y 14 días y horizontes h0 y h7. Conservamos los ocho experimentos aprobados de regresión logística y Random Forest y las referencias de mayoría y persistencia. Enero–marzo de 2026 es evaluación histórica previamente observada.

## Organización técnica

- `config/proyecto.json`: parámetros aprobados y rutas.
- `scripts/` y `tests/`: procesamiento, ajuste, evaluación y verificaciones.
- `data/raw/`, `data/interim/`, `data/processed/`: fuentes y productos locales excluidos de Git.
- `models/` y `results/predictions/`: ocho modelos y predicciones individuales locales excluidos de Git.
- `results/`: métricas, selección congelada, búsquedas y entorno del experimento aprobado.
- `provenance/`: hashes, auditoría y correspondencia de rutas tras la consolidación.
- `reports/INFORME.md` y `reports/figuras/`: resultados oficiales de la Etapa D.
- `specs/`: desarrollo interno SDD, excluido de la entrega y de las exportaciones públicas.

## Reproducción

Usamos Python 3.10.11 y `requirements-lock.txt`. La [documentación técnica](docs/PROCESAMIENTO.md) describe los insumos necesarios, las restricciones de redistribución y los comandos para reproducir el procesamiento y los experimentos.

```powershell
python -m pip install -r requirements-lock.txt
python -m unittest discover -s tests -v
python scripts/verificar.py
python scripts/verificar_integracion.py
```

La integración verifica los datos locales y los resultados sin entrenar nuevos modelos ni sobrescribir artefactos oficiales. El acceso autorizado a los insumos es necesario: los datos individuales no se distribuyen en Git.

## Estado de la comunicación académica

Los notebooks, la metodología, el diccionario, la presentación y el portal conservan su contenido anterior y están pendientes de actualización en la siguiente parte de la Etapa E. Sus cifras anteriores no representan el experimento vigente. La revisión editorial y la publicación se realizarán después de aprobar esta consolidación técnica.

Usamos etiquetas de [Epagri/CIRAM Agroconnect](https://ciram.epagri.sc.gov.br/agroconnect/) y meteorología horaria ERA5 de [Open-Meteo](https://open-meteo.com/), con atribución [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) para los datos meteorológicos. La redistribución de las etiquetas sigue pendiente de aclaración.
