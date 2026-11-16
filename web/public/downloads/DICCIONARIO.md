# Diccionario del archivo largo

| columna | tipo | unidad | descripcion | uso_modelo |
| --- | --- | --- | --- | --- |
| id_medicion | object | — | Identificador estable del objetivo; agrupar por él | auxiliar, excluido |
| latitud | float64 | grados | Latitud WGS84 del monitoreo | auxiliar, excluido |
| longitud | float64 | grados | Longitud WGS84 del monitoreo | auxiliar, excluido |
| fecha_objetivo | object | — | Fecha del resultado publicado | auxiliar, excluido |
| resultado_objetivo | object | — | Etiqueta CIRAM, Positivo o Negativo | etiqueta |
| dias_anteriores | int64 | — | Distancia entera al objetivo: 1 a 14 | auxiliar, excluido |
| fecha_dato | object | — | Fecha local anterior al objetivo | auxiliar, excluido |
| roya_anterior | object | — | Positivo/Negativo auxiliar; vacío si no evaluado o sin registro | auxiliar, excluido |
| estado_registro_roya | object | — | evaluado, no_evaluado o sin_registro | auxiliar, excluido |
| hr_media_pct | float64 | % | hr media pct | resumen de ventana |
| hr_min_pct | float64 | % | hr min pct | resumen de ventana |
| hr_max_pct | float64 | % | hr max pct | resumen de ventana |
| temperatura_min_c | float64 | °C | temperatura min c | resumen de ventana |
| temperatura_max_c | float64 | °C | temperatura max c | resumen de ventana |
| temperatura_media_c | float64 | °C | temperatura media c | resumen de ventana |
| precipitacion_mm | float64 | mm | precipitacion mm | resumen de ventana |
| lluvia_mm | float64 | mm | lluvia mm | auxiliar, excluido |
| viento_vel_media_10m_ms | float64 | m/s | viento vel media 10m ms | resumen de ventana |
| viento_vel_max_10m_ms | float64 | m/s | viento vel max 10m ms | auxiliar, excluido |
| viento_dir_10m_grados | float64 | grados | Dirección vectorial de procedencia; vacío si vector medio nulo | auxiliar, excluido |
| viento_u_media_10m_ms | float64 | m/s | viento u media 10m ms | resumen de ventana |
| viento_v_media_10m_ms | float64 | m/s | viento v media 10m ms | resumen de ventana |
| iith_h | int64 | h | Horas HR>=85 y 15<=T<=28, límites inclusivos | variante IITH |
| horas_hr_ge90_h | int64 | h | Horas HR>=90; auxiliar, no HMF | auxiliar, excluido |

En features_7dias y features_14dias se agregan medias como medias de días completos; mínimos/máximos como extremos; precipitación e IITH como sumas. objetivo_positivo: 1=Positivo, 0=Negativo. municipio y punto_id solo identifican. El archivo procedencia.csv conserva la celda ERA5, zona horaria y pendientes.
