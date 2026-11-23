# Diccionario de las muestras

Cada fila de una tabla de entrenamiento representa una medición con un horizonte y una ventana. Las ventanas de 7 y 14 días comparten las mismas muestras dentro de cada horizonte.

| Campo | Significado |
|---|---|
| `muestra_id`, `observacion_id`, `observacion_objetivo_id` | Identificadores de muestra y de las observaciones de referencia y objetivo. |
| `ubicacion_id`, `punto_id`, `punto_objetivo_id` | Agrupación provisional por coordenadas y códigos de punto originales. La identidad física entre visitas tiene incertidumbre. |
| `fila_fuente`, `fila_objetivo_fuente` | Filas de origen para rastrear el emparejamiento. |
| `fecha_referencia`, `fecha_objetivo` | Fechas de registro. La coincidencia con la primera consulta no acredita cuándo se tomaron las muestras. |
| `y`, `resultado_t` | Etiqueta objetivo y estado de referencia. Positivo = 1 y Negativo = 0. Não Avaliado indica ausencia de evaluación y se excluye como etiqueta. |
| `horizonte_dias` | 0 para el estado en la fecha de referencia y 7 para el estado registrado exactamente siete días después en la misma ubicación. |
| `ventana_dias`, `ventana_inicio`, `ventana_fin` | Intervalo meteorológico de 7 o 14 días que termina el día anterior a la referencia. |
| `hr_media_pct`, `hr_min_pct`, `hr_max_pct` | Humedad relativa media, mínima y máxima de la ventana (%). |
| `temperatura_media_c`, `temperatura_min_c`, `temperatura_max_c` | Temperatura media, mínima y máxima de la ventana (°C). |
| `precipitacion_mm` | Precipitación acumulada en la ventana (mm). |
| `viento_vel_media_10m_ms`, `viento_u_media_10m_ms`, `viento_v_media_10m_ms` | Velocidad media y componentes zonal y meridional medias del viento a 10 m (m/s). |

Solo las diez variables meteorológicas entran en los modelos. La referencia de persistencia utiliza el estado registrado en la fecha de referencia. La disponibilidad operacional de los insumos no está acreditada.
