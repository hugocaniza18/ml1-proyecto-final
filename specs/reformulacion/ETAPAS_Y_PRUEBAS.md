# Etapas y verificación

## Estado inicial

Base bdf644e. Rama feat/reformulacion-horizontes. Worktree separado en el espacio de Codex porque Windows rechazó crear la carpeta junto al proyecto en C:/FIUNA.

El README original tiene una modificación local que elimina el código de materia. Se reproduce en esta rama sin confirmar ni revertir el archivo de main.

## Etapas

1. **Rama y especificación.** Revisar SPEC y contrato de datos con el equipo. Etapa aprobada.
2. **Insumos.** Recibir datasets y resultados preliminares, registrar manifiestos y exposición previa de prueba, auditar identidades, fechas, cobertura, licencia y poblaciones. Presentar fechas concretas de folds y cantidades para aprobarlas.
3. **Procesamiento.** Adaptar data_pipeline.py para muestras h0/h7 y ventanas 7/14. Extraer construcción de particiones a una función con contratos explícitos. Guardar tablas nuevas y reportes de exclusión por versión.
4. **Experimentos.** Adaptar experiments.py para ocho configuraciones sin IITH y referencias. Usar las mismas particiones por horizonte. Congelar selección y evaluar el período histórico observado.
5. **Comunicación.** Actualizar tres notebooks, execute_notebooks.py, verify.py, build_reports.py, metodología, diccionario, informe, presentación y portal. Quitar el bloqueo del exportador basado en existencia de ZIP y permitir exportaciones repetibles con destino y versión explícitos.
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
