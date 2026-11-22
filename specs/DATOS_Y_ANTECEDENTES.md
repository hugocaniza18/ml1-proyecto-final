# Contrato de datos y antecedentes

Estado: contrato aprobado el 22 de noviembre de 2026. Insumos incorporados y auditados en provenance/ y results/eda/.

## Recepción

Conservar cada archivo recibido inmutable en almacenamiento local excluido de Git. Crear un manifiesto por versión con nombre, SHA-256, fuente, fecha de recepción, cobertura temporal, unidades, autor de reconstrucción, scripts utilizados, transformaciones y condiciones de redistribución.

Registrar los resultados experimentales preliminares sin IITH como antecedentes, indicando qué conjuntos y métricas de prueba fueron vistos antes. No importar automáticamente sus parámetros como ganadores.

## Tablas canónicas

Observaciones: ubicacion_id, fecha_registro, clase_original, fuente_id y referencia a la fila original. Conservar punto_id, coordenadas y fechas de muestreo/publicación/consulta si existen.

Meteorología: ubicacion_id, fecha/hora local, zona horaria, fuente, celda/estación, unidades y seis series horarias de temperatura, HR, precipitación, lluvia, velocidad y dirección del viento. Lluvia puede conservarse como auxiliar, sin entrar a las diez variables.

Muestras: muestra_id, ubicacion_id, fecha_referencia, fecha_objetivo, horizonte_dias, ventana_dias, ventana_inicio, ventana_fin, y, resultado_t, las diez variables y estado de disponibilidad. Identidad única por ubicación, referencia, horizonte y ventana. Identificadores solo vinculan registros.

Predictores exactos: hr_media_pct, hr_min_pct, hr_max_pct, temperatura_media_c, temperatura_min_c, temperatura_max_c, precipitacion_mm, viento_vel_media_10m_ms, viento_u_media_10m_ms, viento_v_media_10m_ms.

Calcular medias sobre horas válidas completas, extremos sobre toda la ventana y precipitación como suma. Viento u=−velocidad·sen(dirección), v=−velocidad·cos(dirección), usando dirección de procedencia en grados. Mantener precisión completa. Exigir 24 horas válidas por día local; cualquier excepción de zona horaria debe resolverse explícitamente en auditoría.

## Auditoría

- No fijar como requisito 428 muestras o 5992 filas. Derivar cantidades del manifiesto.
- Revisar duplicados de ubicación/fecha, conflictos de clase, calendarios e intervalos sin seguimiento.
- Resolver la correspondencia de punto_id 14 y 20 con coordenadas compartidas. No asumir equivalencia física sin evidencia.
- Si el dataset recibido es agregado, verificar o documentar su regla de agregación y disponibilidad de series originales.
- Reportar muestras h7 descartadas por ausencia exacta de t+7, no evaluados, conflictos y meteorología incompleta.
- No imputar etiquetas ni arrastrar resultados previos. Los extremos meteorológicos válidos se conservan.

## Antecedentes experimentales

Solicitar configuración, versión de datos, particiones, semilla, hiperparámetros, métricas y predicciones cuando estén disponibles. Recalcular métricas desde las predicciones y comprobar IDs y fechas.

Separar en los reportes los resultados recibidos de los reproducidos con el contrato aprobado. La ausencia de trazabilidad se informa. Recibir un archivo no autoriza ejecutar entrenamiento ni analizar una posible prueba inédita.

## Redistribución

Los datos Open-Meteo requieren atribución CC BY 4.0 y descripción de las transformaciones. Fuente: https://open-meteo.com/en/terms.

La normativa de Epagri restringe el traspaso de datos suministrados a terceros y exige citar su banco de datos. Debe confirmarse su alcance para las etiquetas del endpoint público específico de Agroconnect. Fuente: https://ciram.epagri.sc.gov.br/ciram_arquivos/site/RN_ProcedimentosFornecimentoDadosLaudos.pdf, sección de solicitudes.

El manifiesto distinguirá licencia comprobada, permiso específico y condición no confirmada. Revisar los archivos, notebooks, modelos, predicciones y ejemplos antes de publicar. Hasta aclarar derechos, mantener datos identificables locales y exportar únicamente contenido público revisado. Esta condición no impide realizar la auditoría local.

## Consolidación técnica aprobada

Los ZIP preliminares y sus extracciones duplicadas se retiran de la estructura oficial una vez verificada la preservación de los insumos necesarios. Sus hashes y procedencia quedan registrados. Los archivos recibidos en Downloads no se modifican. Los productos definitivos, sus entradas originales necesarias y sus evidencias se conservan localmente.
