# Resultado de la Etapa C

Construimos h0 y h7, con ventanas 7 y 14, desde insumos locales de Etapa B. Los originales y resultados anteriores se conservan. No se ejecutaron entrenamientos.

| Horizonte | Población por ventana | Desarrollo N/P | Purga N/P | Histórico N/P |
|---|---:|---|---|---|
| h0 | 428 | 188/171 (359) | 0/3 | 29/37 (66) |
| h7 | 362 | 157/156 (313) | 0/3 | 22/24 (46) |

La tabla conjunta contiene 1580 filas, por la repetición de las 790 muestras de ambos horizontes en dos ventanas. Cada modelo futuro utilizará una sola tabla de horizonte y ventana.

| Horizonte | Fold | Entrenamiento | Validación | N/P validación |
|---|---:|---:|---:|---|
| h0 | 1 | 164 | 38 | 23/15 |
| h0 | 2 | 188 | 55 | 29/26 |
| h0 | 3 | 239 | 100 | 56/44 |
| h7 | 1 | 141 | 31 | 19/12 |
| h7 | 2 | 150 | 51 | 25/26 |
| h7 | 3 | 196 | 90 | 51/39 |

Las seis validaciones tienen fechas semanales continuas en el conjunto del bloque, con ambas clases. La cobertura por ubicación puede ser parcial. El resumen de particiones registra los intervalos exactos y la purga aprobada respecto de referencias.

Registramos 510 exclusiones horizonte/observación: 222 referencias no evaluadas para cada horizonte (444), 56 objetivos h7 no evaluados y 10 objetivos h7 sin registro exacto. No hubo pérdida adicional por meteorología ni conflictos en estos insumos.

Pasaron siete pruebas de contrato y las verificaciones de integración. Se probaron cruce de año, emparejamiento exacto, visitas tardías, no evaluados, conflictos, cobertura incompleta, valores no finitos, duplicados meteorológicos, inclusión de clima en fecha de referencia y etiquetas en el límite de purga.

Se verificaron las cuatro tablas contra los antecedentes recibidos con tolerancia absoluta 10⁻¹⁰ en predictores. Las etiquetas y cantidades son idénticas. Una ejecución independiente produjo exactamente los mismos hashes de salida. La escritura sobre un destino ocupado fue rechazada. Los hashes de las fuentes permanecieron iguales.

La disponibilidad operacional sigue no acreditada, por falta de fechas verificadas de muestreo/publicación y por el carácter retrospectivo de ERA5. Los datos y metadatos identificables permanecen locales y excluidos de Git. Etapa C cerrada tras aprobar las verificaciones y correcciones de identidad y trazabilidad.

La prueba adicional distingue coordenadas que solo difieren después de diez cifras significativas e incluso más allá de la precisión float. Se normalizan ceros y representaciones decimales equivalentes, rechazando valores no finitos. C02 y C01 producen las mismas muestras actuales. La reconstrucción horaria verificó 66 respuestas, 177264 horas, 5992 filas diarias y 1580 ventanas, con errores absolutos máximos de 1,42×10⁻¹⁴ y 5,68×10⁻¹⁴. Evidencias en trazabilidad_era5_resumen.json y traza local.
