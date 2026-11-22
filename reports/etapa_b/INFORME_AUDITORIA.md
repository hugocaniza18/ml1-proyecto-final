# Auditoría de insumos y propuesta de particiones

Etapa B, 22 de noviembre de 2026. Especificaciones aprobadas por el equipo. Particiones propuestas pendientes de aprobación antes de Etapa C.

## Recepción y reproducibilidad

Incorporamos copias byte a byte de los dos ZIP y extrajimos sus miembros en almacenamiento local excluido de Git, bajo data/raw/etapa_b_20261122. Los archivos originales de Descargas permanecen intactos. No ejecutamos el script de experimentos recibido ni entrenamientos nuevos.

| Archivo | SHA-256 |
|---|---|
| royasoja_dataset_sin_iith.zip | 170c7a34665d323bf46099d2bf10fadb6c75fffb6a2478bec758daacfc2299ab |
| resultados_experimentos_royasoja_sin_iith.zip | d5d8184cced5b0fd56f9d229c95b47b2e7db55885fb442f239948f1cf319b474 |

El manifiesto procedencia_y_hashes.json registra hashes de cada miembro y de los respaldos. El manifiesto local agrega rutas de origen. Los cuatro archivos del dataset incluidos en ambos ZIP son idénticos por hash.

El script scripts/auditar_etapa_b.py reproduce las verificaciones y reportes desde los ZIP y el paquete original, con pandas, NumPy, scikit-learn (solo métricas) y Matplotlib. Las versiones principales están fijadas en requirements-lock.txt. El script requiere --dataset-zip, --results-zip y --original-root. No descarga datos.

Para reproducibilidad académica conservamos originales, caché horaria y manifiestos localmente. El repositorio puede distribuir el código de auditoría, contratos, hashes y reportes agregados. Un evaluador con autorización necesita aportar los mismos insumos y verificar sus hashes. Un clon público por sí solo no reproduce la auditoría completa. Los IDs de las particiones propuestas y las exclusiones por observación permanecen locales. No incluimos los CSV recibidos ni sus predicciones identificables en las descargas.

La meteorología Open-Meteo requiere atribución CC BY 4.0 y descripción de las transformaciones. La normativa Epagri restringe el traspaso de datos suministrados a terceros; falta confirmar su alcance sobre el endpoint fitosanitario público. Conservamos esa incertidumbre sin atribuir una licencia abierta a las etiquetas. Fuentes: https://open-meteo.com/en/terms y https://ciram.epagri.sc.gov.br/ciram_arquivos/site/RN_ProcedimentosFornecimentoDadosLaudos.pdf.

## Datos, etiquetas y cobertura

El original contiene 650 registros: 211 Positivo (79), 217 Negativo (80) y 222 Não Avaliado (82). No hay duplicados coordenada/fecha ni conflictos marcados. El paquete reconstruido contiene 428 muestras, 31 columnas y diez predictores por ventana. Todas las variables meteorológicas son finitas. No encontramos columnas IITH.

Las poblaciones verificadas son h0 con 428 muestras (217 negativas y 211 positivas) y h7 con 362 (179 negativas futuras y 183 positivas futuras). De las 428 referencias, h7 excluye 56 por resultado no evaluado y 10 por ausencia exacta de registro en t+7. No convertimos ausencias en negativos. El subconjunto auxiliar de referencias inicialmente negativas contiene 188 pares, con 57 positivos y 131 negativos futuros.

Reconstruimos las etiquetas mediante el original por coordenadas y fecha exacta. Los 362 pares conservan el mismo punto_id entre t y t+7. Verificamos fecha_dia_mas_7=t+7 incluso en las filas sin resultado futuro. En el dataset recibido, fecha_objetivo es la fecha de referencia t; para h7 debe utilizarse fecha_dia_mas_7 como fecha de etiqueta.

Los diez predictores por ventana coinciden con las tablas locales de 7 y 14 días con diferencia máxima 3,55×10⁻¹⁵. También reconstruimos las agregaciones desde el detalle de 5992 días, catorce por objetivo. Pasan las comprobaciones de HR en [0,100], temperatura mínima ≤ media ≤ máxima, precipitación/velocidad no negativas y módulo del viento vectorial ≤ velocidad media.

Las 66 respuestas horarias de respaldo pasan integridad de timestamps, longitudes, valores presentes, UTC−3 y viento en m/s. Registramos sus hashes y cobertura. El detalle diario demuestra ventanas t−14 a t−1. La verificación de agregación parte de los diarios existentes; esta etapa no rehace la extracción original desde el portal.

Hay 22 coordenadas y 23 punto_id. Los IDs 14 y 20 comparten coordenada y municipio Jupiá, en períodos distintos. No hay pares h7 que crucen esos IDs. Para esta auditoría contamos ubicaciones por coordenada y conservamos ambos IDs, sin afirmar identidad física de colector o parcela. Las fechas disponibles terminan el 02/03/2026; no recibimos un período nuevo para prueba inédita.

## Significado temporal y disponibilidad

En las 650 filas fecha_registro coincide exactamente con primera_consulta. n_snapshots varía entre 7 y 207. Estos campos describen el seguimiento de consultas y no acreditan fecha de muestreo. Los ZIP reproducen esta advertencia y no aportan diccionario oficial, registros de laboratorio ni el código que reconstruyó el CSV original desde snapshots.

La documentación pública de Epagri describe visitas semanales, análisis de colectores y observación de síntomas, pero no vincula cada etiqueta de este endpoint con un protocolo concreto. No podemos determinar si un positivo corresponde específicamente a esporos, síntomas, primera aparición o persistencia. Fuente: https://blog.epagri.sc.gov.br/alertas-emitidos-pela-epagri-ajudam-no-controle-de-doenca-que-pode-dizimar-lavouras-de-soja/.

La alineación de fechas y la separación cronológica se pueden verificar. La disponibilidad operacional permanece no acreditada: faltan horas de publicación de etiquetas, fecha real de muestreo y versión del clima disponible al emitir la predicción. ERA5 es reanálisis retrospectivo. Persistencia h7 solo simula una regla con y(t) conocido; no se ha demostrado que ese estado estuviera publicado en el instante t.

## Cortes históricos propuestos

Clasificamos la evaluación por fecha de etiqueta dentro de enero–marzo de 2026. Aplicamos la regla aprobada de entrenamiento con fecha objetivo estrictamente anterior a primera referencia de evaluación menos 14 días.

| Horizonte | Desarrollo N/P | Purga N/P | Evaluación N/P | Referencias de evaluación | Etiquetas de evaluación |
|---|---|---|---|---|---|
| h0 | 188/171 (359) | 0/3 | 29/37 (66) | 05/01–02/03/2026 | 05/01–02/03/2026 |
| h7 | 157/156 (313) | 0/3 | 22/24 (46) | 05/01–23/02/2026 | 12/01–02/03/2026 |

N/P representa negativos/positivos. Desarrollo h0 abarca 02/09/2024–15/12/2025, con 22 ubicaciones; h7 abarca referencias 02/09/2024–08/12/2025, también con 22. Las etiquetas de ambos terminan el 15/12/2025. Evaluación h0 incluye 12 ubicaciones, 9 fechas y 56 días calendario de diferencia (57 inclusivos). H7 incluye 11 ubicaciones, 8 fechas y 49 días de diferencia (50 inclusivos).

La purga h0 retira referencias y etiquetas del 22/12/2025. En h7 retira referencias del 15/12/2025 con etiqueta del 22/12/2025. No hay pares disponibles con referencia 29/12/2025 que aporten etiquetas al 05/01/2026.

## Folds propuestos

Cada validación reúne siete fechas observadas. Duración representa fin−inicio en días calendario. El CSV folds_propuestos.csv incluye también los días inclusivos, intervalos meteorológicos, purgas y número de IDs.

| h | Fold | Referencias entrenamiento | n; N/P; ubic. | Referencias validación | Duración | n; N/P; ubic. |
|---|---|---|---|---|---:|---|
| 0 | 1 | 02/09–02/12/2024 | 133; 60/73; 17 | 23/12/2024–08/09/2025 | 259 | 27; 17/10; 10 |
| 0 | 2 | 02/09/2024–25/08/2025 | 174; 85/89; 17 | 15/09–27/10/2025 | 42 | 62; 30/32; 11 |
| 0 | 3 | 02/09/2024–13/10/2025 | 229; 114/115; 17 | 03/11–15/12/2025 | 42 | 109; 63/46; 19 |
| 7 | 1 | 02/09–04/11/2024 | 74; 19/55; 8 | 02/12/2024–01/09/2025 | 273 | 43; 28/15; 17 |
| 7 | 2 | 02/09/2024–11/08/2025 | 143; 64/79; 17 | 08/09–20/10/2025 | 42 | 59; 27/32; 10 |
| 7 | 3 | 02/09/2024–29/09/2025 | 188; 87/101; 17 | 27/10–08/12/2025 | 42 | 98; 57/41; 19 |

En h0 las fechas de etiqueta coinciden con referencia. En h7 se desplazan +7 días: las validaciones de los folds 1, 2 y 3 tienen etiquetas 09/12/2024–08/09/2025, 15/09–27/10/2025 y 03/11–15/12/2025. Las etiquetas de entrenamiento terminan respectivamente 11/11/2024, 18/08/2025 y 06/10/2025.

Todos los bloques contienen ambas clases y cumplen la purga. Los primeros folds abarcan un hueco prolongado entre campañas. El entrenamiento h7 inicial tiene solo ocho ubicaciones y 19 negativos. Estos factores limitan estabilidad y comparabilidad entre folds; no los corregimos buscando mejores métricas. Los bloques son comunes a las dos ventanas dentro de cada horizonte.

## EDA de desarrollo previo al modelado

Generamos descriptivos, correlaciones, cobertura mensual, clases y conteos de extremos por IQR usando exclusivamente desarrollo propuesto. Los gráficos eda_desarrollo_h0.png y eda_desarrollo_h7.png resumen clases y fechas. No eliminamos extremos ni ajustamos preprocesadores.

H0 tiene 188 negativos y 171 positivos. H7 tiene 157 negativos y 156 positivos, aunque el equilibrio global no se conserva en cada fold. Las medianas de temperatura media para ventanas 7/14 son 19,28/19,13 °C en h0 y 19,13/18,73 °C en h7. Las medianas de precipitación acumulada son 35,9/79,1 mm y 33,1/76,6 mm respectivamente. El incremento del acumulado depende de la duración y no demuestra mejora predictiva.

La cobertura mensual tiene interrupciones y cantidades desiguales. La comparación de horizontes usa poblaciones distintas. Sus futuras diferencias de rendimiento deben interpretarse considerando esa composición.

## Auditoría de antecedentes experimentales

Recalculamos las siete métricas de prueba de las ocho configuraciones recibidas desde las 448 predicciones (4×66 h0 y 4×46 h7). Todas coinciden con comparacion_modelos.csv dentro de tolerancia 10⁻¹². Verificamos IDs, etiquetas, fechas objetivo, puntuaciones y umbral. Los resultados históricos ya fueron observados y se registran como exposición previa.

La validación preliminar usa bloques de seis fechas y los últimos 18 días observados, mientras la especificación aprobada propone siete fechas y 21 fechas en total. Para h7, el script preliminar purga respecto del inicio de fechas de etiqueta; el contrato nuevo usa inicio de referencias. Por tanto, no reutilizaremos esos folds como particiones aprobadas.

RF preliminar usa 150 árboles y mínimo por hoja {3,5}; el contrato aprobado usa 300 y {3,5,10}. No recibimos modelos guardados, todas las predicciones de validación, la búsqueda completa ni un registro del entorno de ejecución. Verificamos las métricas finales, pero no podemos acreditar desde esos archivos el ajuste ni reproducir el F1 de validación sin nuevos entrenamientos. El script recibido se conserva inmutable como antecedente.

## Condiciones para Etapa C

Solicitamos aprobación de las poblaciones y cortes propuestos, incluidos los huecos del primer fold y la regla de ubicación por coordenadas con punto_id conservado. Quedan abiertas la identidad física de Jupiá y las fechas reales de muestreo/publicación. El trabajo puede continuar como experimento retrospectivo documentando esas incertidumbres. Una afirmación operacional requerirá evidencia adicional.

No realizamos cambios al portal, nuevos entrenamientos, commits en main ni publicación de los datos recibidos.
