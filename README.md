# Predicción de roya de soja con datos climáticos

Trabajo final de Machine Learning 1 (23433), Facultad de Ingeniería, Universidad Nacional de Asunción, 2026.

**Integrantes:** [Hugo Rene Caniza Portillo](https://github.com/hugocaniza18) y [Ruben Dario Lopez Sanabria](https://github.com/Rub3nL0pez).

Evaluamos si la humedad, la temperatura, la precipitación y el viento de los 7 o 14 días anteriores permiten clasificar el resultado de una medición de roya de soja. En una variante incorporamos el índice térmico-humedad (IITH). El modelo devuelve una puntuación para la clase positiva y una clasificación con umbral de 0,5.

## Resultados

Analizamos 428 mediciones de 22 puntos. La ventana de 14 días reúne 5.992 filas diarias y cada medición aporta un solo objetivo al modelo. Comparamos regresión logística y Random Forest con ambas ventanas, con y sin IITH, usando las mismas particiones temporales.

Seleccionamos Random Forest de 7 días sin IITH por su F1 medio de validación (0,653). En las 66 mediciones reservadas para prueba, el F1 fue 0,473 y el modelo detectó 13 de los 37 positivos. Extender la ventana a 14 días no mejoró el F1 de forma consistente. El estudio evalúa resultados históricos y deja abierta la validación en nuevas campañas.

## Materiales

- [Informe](reports/INFORME.md)
- [Presentación](presentation/Presentacion_TP_Roya.pptx)
- [Metodología](docs/METODOLOGIA.md) y [diccionario de datos](docs/DICCIONARIO.md)
- [Notebooks de auditoría, exploración y modelado](notebooks/)
- [Métricas de los ocho experimentos](reports/comparacion_modelos.csv)

En el portal incluimos los gráficos, un comparador y seis ejemplos de prueba sin identificadores de origen. El portal se puede ejecutar con Node.js 24 mediante `npm ci` y `npm run dev` desde `web/`. Para reproducir los experimentos en Python 3.10, los scripts requieren los archivos originales y la caché meteorológica del paquete local.

## Datos

Usamos los registros de [Epagri/CIRAM Agroconnect](https://ciram.epagri.sc.gov.br/agroconnect/) como etiquetas y datos horarios ERA5 de [Open-Meteo](https://open-meteo.com/) bajo [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). A partir de estos datos calculamos los resúmenes diarios y por ventana, el IITH y las componentes del viento.
