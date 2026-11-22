# Etapas y verificación

## Estado inicial

Base bdf644e. Rama feat/reformulacion-horizontes. Worktree separado en el espacio de Codex porque Windows rechazó crear la carpeta junto al proyecto en C:/FIUNA.

El README original tiene una modificación local que elimina el código de materia. Se reproduce en esta rama sin confirmar ni revertir el archivo de main.

## Etapas

1. **Rama y especificación.** Revisar SPEC y contrato de datos con el equipo. Etapa aprobada.
2. **Insumos.** Recibir datasets y resultados preliminares, registrar manifiestos y exposición previa de prueba, auditar identidades, fechas, cobertura, licencia y poblaciones. Presentar fechas concretas de folds y cantidades para aprobarlas.
3. **Procesamiento.** Adaptar procesar_datos.py para muestras h0/h7 y ventanas 7/14. Extraer construcción de particiones a una función con contratos explícitos. Guardar tablas nuevas y reportes de exclusión por versión.
4. **Experimentos.** Adaptar entrenar.py y evaluar.py para ocho configuraciones sin IITH y referencias. Usar las mismas particiones por horizonte. Congelar selección y evaluar el período histórico observado.
5. **Comunicación.** Actualizar tres notebooks, ejecutar_notebooks.py, verificar.py, generar_informe.py, metodología, diccionario, informe, presentación y portal. Quitar el bloqueo del exportador basado en existencia de ZIP y permitir exportaciones repetibles con destino y versión explícitos.
6. **Revisión de rama.** Ejecutar controles, revisar contenido publicable y preparar PR. La integración y el despliegue de producción se someten a revisión posterior.

## Pruebas

- Datos sintéticos mínimos para pares exactos t+7, falta de visita, Não Avaliado, conflictos, duplicados y cambio de año.
- h7 usa meteorología anterior a t aun cuando haya clima disponible entre t y t+7.
- Lista exacta de diez predictores y ausencia de columnas IITH en entradas/exportaciones nuevas.
- Población coincidente entre ventanas de cada horizonte y trazabilidad de todas las exclusiones.
- Entrenamiento con fecha objetivo menor que inicio de validación menos 14 días, ambas clases por fold y separación de ubicaciones de una misma fecha por bloque.
- Reporte de disponibilidad desconocida o posterior a emisión, sin ocultarla.
- Mayoría calculada solo en entrenamiento y persistencia solo en h7, sin pasar y(t) a LR o RF.
- Métricas recalculadas desde predicciones, selección por CV, congelación antes de evaluación histórica y manejo explícito de métricas indefinidas.
- Repetición offline con datos y versiones idénticos. Comparar tablas, particiones, selección y probabilidades dentro de tolerancia numérica documentada.
- Exportación repetible de documentos y ZIP, filtros de horizonte/ventana/modelo/referencia, enlaces y descargas. Compilación, TypeScript y revisión móvil.

## Interfaz de resultados

Exportar versión del experimento, hashes, horizonte, ventana, modelo/referencia, parámetros, fold, muestra/población, métricas y condición de evaluación histórica observada. Los reportes públicos contendrán los resúmenes revisados. team.json seguirá siendo la única fuente del equipo.

## Estado de avance

Etapas A y B aprobadas. Los folds definitivos son tres bloques de seis fechas semanales alineados por etiqueta. Etapa C aprobada con ajustes y cerrada tras verificarlos. Plan detallado de D en ETAPA_D.md, pendiente de revisi?n antes de ejecutar modelos. El nuevo procesamiento est? documentado en docs/PROCESAMIENTO_HORIZONTES.md.

## Condiciones para avanzar

La aprobación general del plan permite esta etapa de especificación. El procesamiento y modelado requieren revisión de estas especificaciones y auditoría de los archivos proporcionados. Los cortes concretos se aprueban antes de entrenar. Los datos verdaderamente nuevos se reservan antes de analizar sus resultados. No se han fijado resultados ni tamaños de muestra futuros.


# Etapa D — entrenamiento y evaluación

Estado: plan aprobado por el equipo. Implementación y experimentos ejecutados; resultados aprobados por el equipo. Evidencia en results/ y productos locales en models/ y results/predictions/.

## 1. Entrada congelada y preparación

Usar data/processed/horizontes_v2_c02 como única versión de muestras y particiones. Verificar hashes del manifiesto y la traza ERA5 antes de ajustar modelos. Registrar commit de implementación, versión de configuración, Python, dependencias y hashes. Elegir un destino local nuevo por ejecución, sin sobrescribir antecedentes ni Etapa C.

Crear una configuración declarativa simple para modelos, grillas, semilla, umbral, features y rutas. El código de entrenamiento no importará el script preliminar. Tomar los diez predictores de predictores.json; rechazar columnas adicionales. resultado_t se entrega exclusivamente al cálculo de persistencia.

Reservar enero–marzo de 2026 como evaluación histórica previamente observada, con los IDs aprobados. La separación se toma de particiones.json, sin volver a elegir fechas según rendimiento. Los resultados preliminares recibidos quedan documentados como antecedentes expuestos, fuera de las entradas de selección.

No se requiere descargar datos. El entorno público necesita acceso autorizado a los insumos locales para reproducir ajustes. Hasta aclarar redistribución, tablas identificables, predicciones por muestra y modelos permanecerán locales.

## 2. Selección exclusivamente en desarrollo

Ocho configuraciones principales: h0/h7 × 7/14 días × LR/RF. Cada configuración usa exactamente la misma población y los mismos folds que las restantes del horizonte.

LR: Pipeline(StandardScaler, LogisticRegression), C=[0.1,1,10], max_iter=3000, random_state=42. RF: 300 árboles, max_depth=[3,5,None], min_samples_leaf=[3,5,10], random_state=42. Mantener los demás parámetros predeterminados de la versión fijada. GridSearchCV con las tres particiones explícitas, scoring F1 positivo y error_score='raise'. Sin escalado previo al pipeline. Umbral 0.5.

Para empates exactos entre parámetros de la misma configuración, elegir primero el orden de grilla declarado: C ascendente; profundidad 3,5,None; mínimo por hoja 3,5,10. Guardar la tabla completa de candidatos y F1 por fold. Entre configuraciones, seleccionar por horizonte según F1 medio, redondeado a tres decimales para desempatar, favoreciendo 7 días y luego LR.

En cada fold, ajustar la referencia mayoritaria solo con entrenamiento. En empate de clases elegir 0; puntuación constante igual a prevalencia positiva de entrenamiento. Para h7, persistencia predice resultado_t y usa esa clase 0/1 como puntuación. Calcular referencias una vez por horizonte y fold, compartidas entre ventanas. Persistencia requiere y(t) conocido bajo la simulación; conservar la disponibilidad operacional como no acreditada.

Guardar predicciones de validación de los candidatos seleccionados por configuración. Si GridSearchCV no conserva las predicciones, repetir exclusivamente esos ajustes por fold para exportarlas. Registrar que CV intervino en ajuste y selección, por lo que su resultado no es una evaluación independiente.

No aplicar remuestreo, ponderación de clases, calibración ni ajuste de umbral en esta versión. No añadir estado previo al clasificador meteorológico.

## 3. Congelación y evaluación histórica

Antes de calcular predicciones históricas de la ejecución nueva, guardar la selección por horizonte, parámetros elegidos de las ocho configuraciones, grillas, reglas, hashes de datos y particiones y semilla. Una ejecución con selección congelada distinta debe generar una nueva versión identificada, sin sobrescribir la anterior.

Refit de cada configuración usando únicamente su desarrollo aprobado. Evaluar las ocho configuraciones y las referencias en los mismos IDs históricos de cada horizonte. La mayoría usa exclusivamente desarrollo; persistencia usa resultado_t de las filas históricas. Conservar la selección obtenida en CV aunque otra configuración tenga mejor métrica histórica.

Reportar F1, precisión, recall, accuracy, balanced accuracy, ROC-AUC y average precision. Matriz de confusión con orden [0,1], conteos y normalización por clase real. Usar etiquetas explícitas de positivo/negativo y horizonte. Para denominadores nulos en precisión/recall/F1 usar cero y señalar la ausencia de predicciones positivas; si falta una clase real, ROC-AUC y otras métricas que la requieran quedan no disponibles con motivo. AP y ROC de referencias constantes/binarias se identifican como tales.

Mantener métricas por fold y su media y desviación estándar descriptiva poblacional. No presentar la dispersión de tres folds como intervalo de confianza.

## 4. Diagnósticos y resultados verificables

Comparar LR/RF, 7/14 días y referencias dentro de cada horizonte, indicando tamaños y composición. Calcular diferencias descriptivas de F1 entre ventanas y frente a referencias. Comparaciones entre h0 y h7 deben mostrar que las poblaciones son diferentes.

Generar matrices de confusión, ROC/PR y curvas de aprendizaje temporales. Para estas últimas, usar 40%,70%,100% de fechas completas de cada entrenamiento, con los parámetros ya elegidos y el bloque de validación fijo; omitir prefijos sin ambas clases y registrar motivo. Son diagnósticos, sin nueva selección ni modificación por resultados históricos.

Guardar modelos de las ocho configuraciones con pipeline, orden de features, horizonte, ventana, umbral y hashes. Guardar predicciones por muestra, configuración, partición, resultado real, puntuación y clase, además de métricas, búsqueda completa y selección congelada. Cada muestra histórica aparece una vez por configuración. Para CV puede aparecer en roles distintos entre folds y se identifica por bloque.

Crear resumen agregado para informe futuro con versión, evaluación histórica ya observada y disponibilidad no acreditada. Los notebooks finales, el portal y la presentación se actualizarán en Etapa E, después de revisar los experimentos.

## 5. Verificación y aceptación

- Probar con datos sintéticos que scaler y mayoría se ajustan solo con entrenamiento, que h7 persistencia usa resultado_t y que LR/RF reciben exclusivamente los diez predictores.
- Verificar IDs, cantidades, hashes y separación temporal antes de ajustar. Fallar si las particiones difieren de las aprobadas.
- Probar desempates, métricas indefinidas y conservación de selección al cambiar únicamente resultados históricos en una prueba sintética.
- Recalcular todas las métricas exportadas desde las predicciones. Verificar conteos y matrices.
- Recargar modelos y reproducir puntuaciones históricas con tolerancia absoluta 10⁻¹² dentro del entorno fijado.
- Repetir la ejecución con las mismas entradas y comparar selección, métricas y probabilidades. Los archivos binarios de serialización pueden incluir metadatos; la equivalencia se juzga por predicciones/configuración, no solo por sus bytes.
- Comprobar que no se alteraron originales, antecedentes ni datos de C, y que los productos locales estén excluidos de Git.

Entregar tablas comparativas y diagnósticos de desarrollo e histórico, referencias, modelos y predicciones locales, informe experimental agregado y registro de verificaciones. Presentar los resultados al equipo antes de comenzar Etapa E. La finalización de C y este plan no ejecutan ni aprueban automáticamente ajustes nuevos.
