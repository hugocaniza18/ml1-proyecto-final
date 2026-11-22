# Especificación de la reformulación

Estado: contrato aprobado. Etapas B, C y D aprobadas; consolidación técnica en curso.

## Propósito

Evaluamos la clasificación del resultado registrado por CIRAM en la fecha de referencia y siete días después. El horizonte futuro es experimental. Su interpretación depende del protocolo de las etiquetas y de la fecha real de disponibilidad de los datos.

## Requisitos confirmados

- Dos horizontes, h0 y h7, con fecha objetivo t y t+7 respectivamente.
- Para h7, incluir todas las observaciones evaluadas en t con resultado evaluado exactamente en t+7, en la misma ubicación.
- Dos ventanas de días completos anteriores a t: [t−7,t−1] y [t−14,t−1]. La ventana nunca se desplaza hasta la fecha objetivo de h7.
- Diez predictores: HR media/mínima/máxima, temperatura media/mínima/máxima, precipitación acumulada, velocidad media del viento y componentes medias u/v.
- Excluir IITH de las nuevas tablas experimentales, cálculos, selección, exportaciones y gráficos. Los antecedentes originales se conservan con su versión.
- Evaluar Regresión Logística y Random Forest, más referencias de mayoría y persistencia.
- Mantener enero–marzo de 2026 como evaluación histórica previamente observada.
- Definir una prueba inédita antes de analizar resultados si se reciben datos realmente nuevos. No elegir automáticamente un porcentaje de reserva.

## Población y significado del objetivo

Positivo se codifica como 1 y Negativo como 0. Não Avaliado, falta de registro y conflicto son estados separados, excluidos del objetivo binario.

H0 requiere una evaluación válida en t. H7 requiere evaluaciones válidas en t y exactamente en t+7. No se acepta la visita siguiente como sustitución. Comparar ventanas en la intersección de cobertura completa de 7 y 14 días dentro de cada horizonte. Reportar las poblaciones de cada horizonte por separado.

Las diez variables meteorológicas constituyen las entradas de LR y RF. El resultado en t se permite exclusivamente en la referencia de persistencia h7. La primera detección por campaña queda fuera del objetivo de esta versión.

## Experimentos

LR utiliza StandardScaler ajustado dentro de cada fold, C en {0.1,1,10}, máximo 3000 iteraciones y semilla 42. RF utiliza 300 árboles, profundidad en {3,5,None}, mínimo por hoja en {3,5,10}, semilla 42. Umbral 0.5.

Son ocho configuraciones principales: horizonte × ventana × modelo. Cada horizonte selecciona su configuración por F1 positivo medio en validación. Empates redondeados a tres decimales favorecen ventana de 7 días y luego LR.

La mayoría usa la clase más frecuente del entrenamiento de cada fold; en empate elige 0. Para métricas de puntuación utiliza la fracción de positivos del entrenamiento como puntuación constante. Persistencia h7 predice y(t+7)=y(t), con puntuación binaria 0/1. Se calcula una vez por población y partición, sin duplicarla como experimento de ventana.

Reportar F1, precisión, recall, accuracy, balanced accuracy, ROC-AUC y average precision. Identificar puntuaciones constantes o binarias de las referencias. Si una métrica requiere ambas clases y falta alguna, marcarla como no disponible con el motivo.

## Validación y límites temporales

La evaluación histórica incluye fechas objetivo entre 2026-01-01 y 2026-03-31 inclusive, limitadas a observaciones efectivamente disponibles. Para h7, una referencia de diciembre puede tener etiqueta de enero y pertenece a evaluación histórica.

Desarrollo usa fechas objetivo anteriores al corte, aplicando purga. Para un bloque que empieza en fecha de referencia v, toda fecha objetivo de entrenamiento debe ser estrictamente menor que v−14 días. Comprobar también la separación efectiva de ventanas. Para evaluación histórica, usar como v la primera fecha de referencia de su población y aplicar el mismo criterio. Esto se documenta por horizonte.

Proponer tres validaciones crecientes de siete fechas observadas cada una en desarrollo, con bloques comunes a ambas ventanas y ambos modelos del horizonte. Aprobar las fechas después de auditar los nuevos insumos. Si faltan clases o cobertura, detener la etapa de modelado y presentar el problema al equipo.

Cada fold debe informar fechas inicial/final de referencia, etiquetas y meteorología, duración calendario (fin−inicio y número de días inclusivo), fechas observadas, muestras, clases, ubicaciones y exclusiones por purga.

Registrar las fechas reales de disponibilidad de predictores y resultados cuando existan. En simulación prospectiva, verificar que estén disponibles al emitir la predicción. Cuando se desconozcan, declarar disponibilidad no verificada. ERA5 retrospectivo y fechas de consulta no acreditan disponibilidad operacional.

Congelar selección, configuración y hashes antes de calcular la evaluación histórica nueva. Los resultados históricos ya observados no se emplean para optimizar parámetros, umbral, ventanas ni selección.

## Aceptación

Se requiere trazabilidad desde fuente hasta cada muestra, ausencia de IITH, coincidencia de poblaciones entre ventanas, separación temporal comprobada y métricas reproducibles desde predicciones. El informe y el portal deben identificar la exposición previa de la evaluación de 2026 y el carácter experimental de h7.
