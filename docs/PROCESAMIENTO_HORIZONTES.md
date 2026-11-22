# Procesamiento reproducible de horizontes

## Datos y construcción

Construimos las muestras desde el original de 650 observaciones y el detalle meteorológico diario auditado en Etapa B. Conservamos los ZIP y los respaldos locales sin sobrescribirlos. El procesamiento nuevo está implementado en scripts/procesar_horizontes.py y no depende de los módulos de entrenamiento.

La meteorología de entrada son diarios ya verificados contra el respaldo horario. Esta etapa agrega esos diarios con precisión completa. No vuelve a consultar la API ni reconstruye las series horarias desde Internet. La procedencia de las 24 horas por día permanece en el manifiesto y en la auditoría horaria de Etapa B.

Seleccionamos explícitamente diez predictores del respaldo diario. Las medias se agregan como medias de días completos, los mínimos y máximos como extremos de toda la ventana y la precipitación como suma. Las componentes del viento se promedian en u/v. Las ventanas abarcan t−7 a t−1 y t−14 a t−1.

El identificador de ubicación representa coordenadas normalizadas numéricamente, sin redondear puntos distintos. Leemos latitud y longitud como texto y normalizamos con Decimal exacto y formato fijo, sin límite de cifras significativas. Valores equivalentes como -27.000 y -2.7e1 comparten identidad, mientras dos decimales numéricamente distintos permanecen distintos, incluso si se colapsarían al convertirlos a float. Conservamos punto_id y punto_objetivo_id, incluyendo la incertidumbre sobre Jupiá. observacion_id es un hash estable de ubicación y fecha. muestra_id agrega el horizonte; su combinación con ventana_dias identifica una fila. La misma muestra comparte ID entre ventanas para aplicar particiones idénticas.

En cada fila guardamos los IDs y números de fila de las observaciones de referencia y objetivo, el ID meteorológico anterior, las fechas, la fuente y la ventana. No se utiliza el orden del CSV para emparejar el futuro.

H0 toma la etiqueta en t. H7 requiere etiqueta evaluada exactamente en t+7 en las mismas coordenadas. La selección de ambos horizontes parte de todas las observaciones del original, y documenta por separado referencias no evaluadas, conflictos, objetivos faltantes o no evaluados y cobertura meteorológica insuficiente.

## Contratos de salida

La salida local versionada incluye observaciones.csv, muestras.csv, cuatro tablas muestras_h{0,7}_{7,14}d.csv, exclusiones.csv, asignaciones.csv, particiones.json, resumen_particiones.csv, predictores.json y manifiesto.json.

predictores.json es la lista autorizada de las diez entradas del modelo y sus unidades. resultado_t se conserva para la referencia de persistencia posterior. Los IDs, etiquetas, fechas y metadatos no son predictores meteorológicos.

Los valores de fecha_registro se preservan con su semántica incierta. En esta versión no hay fechas acreditadas de toma de muestra ni de publicación; disponibilidad se registra como no_acreditada. No se interpreta primera_consulta como prueba de disponibilidad de muestreo. La separación cronológica puede verificarse, pero la disponibilidad operacional sigue pendiente.

## Particiones aprobadas

Los tres bloques contienen seis fechas semanales de etiqueta cada uno:

| Bloque | Etiquetas de validación | Referencias h0 | Referencias h7 |
|---|---|---|---|
| 1 | 18/08–22/09/2025 | 18/08–22/09/2025 | 11/08–15/09/2025 |
| 2 | 29/09–03/11/2025 | 29/09–03/11/2025 | 22/09–27/10/2025 |
| 3 | 10/11–15/12/2025 | 10/11–15/12/2025 | 03/11–08/12/2025 |

Cada entrenamiento admite únicamente etiquetas estrictamente anteriores a primera referencia de validación menos 14 días. Comprobamos además separación efectiva de ventanas meteorológicas y presencia de ambas clases.

La evaluación histórica se selecciona por etiqueta dentro de enero–marzo de 2026. El corte de entrenamiento se calcula desde su primera referencia y con la misma purga. Los datos fuera del período aprobado provocan una revisión explícita. Las observaciones de una misma fecha de referencia tienen la misma pertenencia temporal.

Cada partición guarda IDs por horizonte, compartidos entre ventanas. Su resumen incluye fechas iniciales y finales de referencia, etiquetas y clima, días transcurridos e inclusivos, fechas observadas, muestras, clases, ubicaciones, IDs originales y purga.

## Reproducción y controles

Desde la raíz del worktree, con las dependencias Python fijadas y los insumos locales de Etapa B:

```powershell
python scripts/procesar_horizontes.py --output-dir data/processed/horizontes_v2_c02
python -m unittest discover -s tests -v
python scripts/verificar_etapa_c.py --output-dir data/processed/horizontes_v2_c02
```

El primer comando requiere un destino nuevo o vacío para conservar versiones anteriores. Para repetir, elegir otro destino. El verificador realiza una ejecución independiente en un directorio temporal y compara los hashes de todas las salidas. Los resultados son deterministas en el entorno registrado. Los manifiestos registran hashes de fuentes, script, recepción y productos, además de versiones Python, pandas y NumPy.

Las cantidades del dataset aprobado se comprueban en la verificación de integración. La construcción no fija 428 muestras como requisito general. Los errores estructurales, clases desconocidas, duplicados meteorológicos, fechas incompatibles o valores físicamente inválidos detienen el procesamiento. Los datos ausentes y las referencias no utilizables generan exclusiones con motivo explícito.

El procesamiento nuevo tiene un punto de entrada propio para preservar los scripts y resultados históricos. Los módulos anteriores corresponden a la versión previa y no deben usarse para generar estas tablas.

## Reproducibilidad y distribución

Los CSV de observaciones, muestras y exclusiones y los IDs de particiones quedan en data/processed, excluido de Git. Los originales permanecen en data/raw, también excluido. El código, las pruebas y el resumen agregado pueden acompañar la entrega. Para una reproducción completa, el evaluador debe disponer de los insumos autorizados y verificar sus hashes. El paquete público por sí solo no contiene esos datos.

La atribución meteorológica corresponde a Open-Meteo/ERA5, CC BY 4.0, con agregaciones realizadas para el proyecto. La redistribución de etiquetas Epagri requiere aclarar las condiciones aplicables al endpoint público. Las limitaciones de fecha, protocolo y disponibilidad se conservan en esta versión.

## Cierre y trazabilidad horaria

La revisión final de C incorpora scripts/verificar_traza_era5.py. Verifica los hashes de 66 respuestas contra la evidencia de B y los tres manifiestos de consultas, valida unidades y timestamps horarios completos, reconstruye los diez resúmenes diarios y las ventanas. Se verificaron 177264 horas, 5992 filas diarias y 1580 filas de ventanas; errores máximos absolutos 1,42×10⁻¹⁴ y 5,68×10⁻¹⁴, inferiores a 10⁻¹⁰. No se descargaron datos.

La traza local en data/processed/traza_era5_c02 vincula cada fila diaria con el archivo horario y su SHA-256. traza.json conserva las URL de consulta, hashes de los manifiestos, celdas ERA5, errores por variable y hashes de productos. Las muestras contienen id_medicion_anterior, que permite enlazar sus días con esa traza. Las fechas se asignan a timestamps locales 00–23; la precipitación horaria conserva la asignación de la API a la hora anterior.

Para verificar esta cadena se ejecuta:

```powershell
python scripts/verificar_traza_era5.py --hourly-root ../outputs/roya-soja/data/raw --processed-dir data/processed/horizontes_v2_c02 --output-dir data/processed/traza_era5_c02
```

El destino debe ser nuevo. La ruta hourly-root apunta al respaldo local existente y puede ajustarse al mover el paquete. Mantuvimos la versión c01; c02 incorpora la corrección de identidad. En los datos actuales ambas producen muestras idénticas. Etapa C cerrada con siete pruebas de contrato y verificación de integración superadas. El equipo aprobó el plan de la Etapa D.
