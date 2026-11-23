# Comparación final V1 y V2

Selección de modelos mediante validación temporal sobre desarrollo. Evaluación histórica con umbral 0,5. No se optimizan umbrales en el histórico.

| Horizonte | Experimento | F1 | Precision | Recall | PR AUC | TP | FN |
|---|---|---:|---:|---:|---:|---:|---:|
| h0 | V1 | 0.3846 | 0.6667 | 0.2703 | 0.6947 | 10 | 27 |
| h0 | V2-A | 0.4074 | 0.6471 | 0.2973 | 0.7220 | 11 | 26 |
| h0 | V2-B | 0.2500 | 0.5455 | 0.1622 | 0.7289 | 6 | 31 |
| h7 | V1 | 0.4848 | 0.8889 | 0.3333 | 0.6225 | 8 | 16 |
| h7 | V2-A | 0.4000 | 0.6364 | 0.2917 | 0.6342 | 7 | 17 |
| h7 | V2-B | 0.2143 | 0.7500 | 0.1250 | 0.6487 | 3 | 21 |

## Interpretación

- h0: V2-A eleva levemente F1 frente a V1, pero no demuestra mejora general.
- h7: V1 conserva el mejor F1 frente a V2-A y V2-B.
- V2-B reduce F1 y recall en ambos horizontes.
- La recuperación de positivos es baja en los tres enfoques.
- No atribuir causalidad ni significación estadística a estas diferencias pequeñas.
- El conjunto histórico ya se consultó para comparar V1/V2: no ajustar decisiones basándose en él.
- ERA5 estima condiciones ambientales; no mide mojado foliar.
- La V1 permanece como modelo publicado; V2 se conserva como experimento.
