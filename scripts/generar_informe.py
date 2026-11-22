"""Figuras y reporte experimental a partir de resultados guardados de D."""
from pathlib import Path
import argparse
import json
import shutil
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay, PrecisionRecallDisplay
from procesar_datos import ROOT, sha, require, write_json

plt.rcParams.update({'font.size': 10, 'figure.dpi': 110, 'savefig.dpi': 150,
                     'axes.spines.top': False, 'axes.spines.right': False})


def md_table(frame):
    def value(v):
        return f'{v:.3f}' if isinstance(v, (float, np.floating)) else str(v)
    return '\n'.join(['| ' + ' | '.join(frame.columns) + ' |',
                      '| ' + ' | '.join(['---']*len(frame.columns)) + ' |'] +
                     ['| ' + ' | '.join(value(v) for v in row) + ' |'
                      for row in frame.itertuples(index=False, name=None)])


def report(source, destination):
    require(not destination.exists(), 'Reporte ya existente')
    destination.mkdir(parents=True)
    figures = destination/'figuras'
    figures.mkdir()
    comp = pd.read_csv(source/'comparacion.csv')
    pred = pd.read_csv(source/'predictions/predicciones.csv')
    learn = pd.read_csv(source/'aprendizaje.csv')
    frozen = json.loads((source/'seleccion.json').read_text(encoding='utf-8'))
    verification = json.loads((source/'verificacion.json').read_text(encoding='utf-8'))
    require(verification['repeticion_completa_coincide'] is True, 'Repetición todavía no verificada')
    comparisons = []
    for h in [0, 7]:
        subset = comp[comp.horizonte.eq(h)].sort_values(['ventana', 'config'])
        hist = pred[pred.horizonte.eq(h) & pred.rol.eq('historico')]
        names = subset.config.tolist()
        fig, axes = plt.subplots(2, 3, figsize=(14, 8))
        for ax, name in zip(axes.flat, names):
            group = hist[hist.config.eq(name)]
            ConfusionMatrixDisplay.from_predictions(group.y, group.prediccion, labels=[0, 1],
                display_labels=['Negativo', 'Positivo'], normalize='true', ax=ax,
                cmap='Greens', colorbar=False)
            ax.set_title(name + (' (seleccionado)' if name == frozen['selected'][str(h)] else ''))
        for ax in list(axes.flat)[len(names):]:
            ax.axis('off')
        fig.suptitle(f'h{h} · evaluación histórica observada · proporciones por clase real')
        fig.tight_layout()
        fig.savefig(figures/f'confusion_h{h}.png')
        plt.close(fig)
        # Conteos en archivo separado del gráfico normalizado.
        subset[['config', 'historico_tn', 'historico_fp', 'historico_fn', 'historico_tp']].to_csv(
            destination/f'confusion_conteos_h{h}.csv', index=False)
        for curve, display in [('roc', RocCurveDisplay), ('pr', PrecisionRecallDisplay)]:
            fig, ax = plt.subplots(figsize=(8, 6))
            for name in names:
                group = hist[hist.config.eq(name)]
                display.from_predictions(group.y, group.probabilidad, name=name, ax=ax)
            ax.set_title(f'h{h} · {curve.upper()} · histórico observado')
            if curve == 'pr':
                ax.axhline(hist[hist.config.eq(names[0])].y.mean(), ls='--', color='gray', label='Prevalencia')
            ax.legend(fontsize=8)
            fig.tight_layout()
            fig.savefig(figures/f'{curve}_h{h}.png')
            plt.close(fig)
        fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharey=True)
        models = subset[subset.ventana.gt(0)]
        for ax, name in zip(axes.flat, models.config):
            selected = learn[learn.config.eq(name) & learn.estado.eq('ok')]
            means = selected.groupby('fraccion')[['n', 'train_f1', 'validation_f1']].mean()
            ax.plot(means.n, means.train_f1, 'o-', label='Entrenamiento')
            ax.plot(means.n, means.validation_f1, 'o-', label='Validación')
            ax.set(title=name, xlabel='Mediciones de entrenamiento, media', ylim=(0, 1.05))
            ax.legend(fontsize=8)
        fig.supylabel('F1 positivo')
        fig.tight_layout()
        fig.savefig(figures/f'aprendizaje_h{h}.png')
        plt.close(fig)
        for model in ['LR', 'RF']:
            pair = models[models.config.str.startswith(model+'_')].set_index('ventana')
            comparisons.append(dict(horizonte=h, modelo=model, comparacion='14d menos 7d',
                delta_cv_f1=float(pair.loc[14, 'cv_f1']-pair.loc[7, 'cv_f1']),
                delta_historico_f1=float(pair.loc[14, 'historico_f1']-pair.loc[7, 'historico_f1'])))
        for row in models.itertuples():
            for reference in subset[subset.ventana.eq(0)].itertuples():
                comparisons.append(dict(horizonte=h, modelo=row.config, comparacion='menos '+reference.config,
                    delta_cv_f1=row.cv_f1-reference.cv_f1,
                    delta_historico_f1=row.historico_f1-reference.historico_f1))
    pd.DataFrame(comparisons).to_csv(destination/'diferencias_descriptivas.csv', index=False)
    # No se exportan modelos ni predicciones identificables al reporte público.
    # Las búsquedas oficiales se conservan en results/busquedas.
    table = comp[['config', 'seleccionado', 'cv_f1', 'cv_std', 'historico_f1',
                  'historico_precision', 'historico_recall', 'historico_accuracy',
                  'historico_balanced_accuracy', 'historico_roc_auc', 'historico_average_precision']]
    text = """# Experimentos definitivos de la Etapa D

Machine Learning 1, FIUNA. Hugo Rene Caniza Portillo y Ruben Dario Lopez Sanabria.

## Diseño experimental

Evaluamos dos horizontes, h0 y h7, con ventanas meteorológicas de 7 y 14 días. Utilizamos exclusivamente horizontes_v2_c02, diez predictores y las particiones aprobadas. LR incluye StandardScaler dentro del pipeline. RF utiliza 300 árboles. Mantuvimos semilla 42 y umbral 0,5.

Las entradas son HR media/mínima/máxima, temperatura media/mínima/máxima, precipitación acumulada, velocidad media del viento y componentes u/v. Se excluye IITH. El estado actual solo se utiliza para la referencia de persistencia h7.

GridSearchCV ajustó LR con C en {0,1; 1; 10} y RF con profundidad {3,5,sin límite} y mínimo por hoja {3,5,10}. Usamos tres validaciones crecientes de seis fechas semanales de etiqueta y la purga aprobada respecto de referencias. Seleccionamos por F1 positivo medio dentro de cada horizonte. Los empates redondeados a tres decimales favorecen 7 días y luego LR.

La mayoría se ajusta solo con entrenamiento y elige negativo en empate. Su puntuación es la prevalencia de entrenamiento. Persistencia usa y(t) como clase y puntuación binaria para h7. Las referencias se evalúan una vez por población y partición.

## Poblaciones y evaluación

H0 comprende 428 muestras, con 359 en desarrollo, 3 en purga y 66 en evaluación histórica (29 negativas, 37 positivas). H7 comprende 362, con 313 en desarrollo, 3 en purga y 46 históricas (22 negativas, 24 positivas). Las dos ventanas comparten muestras dentro de cada horizonte.

Las validaciones tienen etiquetas 18/08–22/09/2025, 29/09–03/11/2025 y 10/11–15/12/2025. En h7, las referencias son siete días anteriores. Los tamaños entrenamiento/validación son h0: 164/38, 188/55 y 239/100; h7: 141/31, 150/51 y 196/90. El primer bloque evalúa transferencia desde datos de la campaña anterior.

Enero–marzo de 2026 es una evaluación histórica previamente observada. Conservamos las particiones y la selección de desarrollo. Esta evaluación no constituye una prueba inédita. El historial anterior de experimentos se considera exposición previa, no una fuente para ajustar parámetros en esta ejecución.

## Resultados comparativos

F1, precisión y recall corresponden a la clase positiva. DE es la desviación estándar poblacional entre tres folds, descriptiva. CV participó en ajuste y selección, por lo que su resultado tampoco es una evaluación independiente.

"""
    text += md_table(table) + '\n\n'
    text += '## Modelos seleccionados exclusivamente en validación\n\n'
    for h in [0, 7]:
        name = frozen['selected'][str(h)]
        row = comp[comp.config.eq(name)].iloc[0]
        cfg = next(r for r in frozen['configs'] if r['config'] == name)
        text += (f"Para h{h} seleccionamos **{name}**, parámetros {cfg['parametros']}, "
                 f"F1 CV {row.cv_f1:.3f} ± {row.cv_std:.3f}. En histórico obtuvo F1 {row.historico_f1:.3f}, "
                 f"precisión {row.historico_precision:.3f}, recall {row.historico_recall:.3f}, "
                 f"ROC-AUC {row.historico_roc_auc:.3f} y AP {row.historico_average_precision:.3f}. "
                 f"Reconoció {int(row.historico_tp)} positivos y omitió {int(row.historico_fn)}.\n\n")
    text += """## Interpretación

En h0, RF de 7 días supera ligeramente a LR de 14 días en F1 de validación; el modelo elegido alcanza F1 histórico 0,385 y reconoce 10 de 37 positivos. La referencia mayoritaria elige negativo en desarrollo y tiene F1 positivo histórico cero. Superarla no establece utilidad agronómica.

En h7, LR de 14 días alcanza F1 histórico 0,485 y reconoce 8 de 24 positivos. Persistencia obtiene F1 0,735, recall 0,750 y accuracy 0,717, frente a 0,630 de accuracy del modelo climático elegido. La comparación favorece la referencia en ese período histórico. Persistencia utiliza estado actual, por lo que tiene información diferente de los clasificadores meteorológicos.

La mayoría h7 tiene F1 medio de validación 0,613 y F1 histórico cero. El cambio de clase mayoritaria entre entrenamientos muestra por qué accuracy o un F1 medio aislado requieren examinar cada fold. Las predicciones de mayoría en empate se evalúan según la regla de clase 0, aunque la puntuación sea 0,5.

Extender la ventana favorece a LR en F1 de validación en ambos horizontes, mientras RF obtiene mayor F1 con 7 días. No aparece una mejora uniforme para todos los algoritmos. Los deltas completos se incluyen en diferencias_descriptivas.csv. Estos resultados son descriptivos y no prueban significancia estadística.

El ROC-AUC histórico 0,464 de LR de 14 días h7 advierte que su ordenamiento global de puntuaciones es débil en esa población, aunque a umbral 0,5 sus nueve positivos predichos incluyan ocho aciertos. No ajustamos el umbral ni calibramos probabilidades después de observar este comportamiento.

## Curvas y matrices

Las matrices se normalizan por clase real; sus conteos se guardan en CSV. ROC y PR utilizan las puntuaciones originales. Las curvas de las referencias corresponden a puntuaciones constantes o binarias. La línea de prevalencia en PR describe la proporción positiva de la evaluación.

Las curvas de aprendizaje usan prefijos cronológicos de fechas completas al 40 %, 70 % y 100 % del entrenamiento de cada fold, con hiperparámetros ya elegidos. Los puntos representan promedios de bloques. Son diagnósticos de desarrollo y no modificaron la selección.

"""
    for h in [0, 7]:
        for kind in ['confusion', 'roc', 'pr', 'aprendizaje']:
            text += f'![{kind}, horizonte {h}](figuras/{kind}_h{h}.png)\n\n'
    text += """## Limitaciones y conclusiones

La etiqueta representa el resultado registrado. fecha_registro coincide con primera_consulta en los originales y no acredita fecha de muestreo. Faltan datos de publicación para confirmar que cada etiqueta estaba disponible al emitir una predicción. El origen específico de cada positivo —esporos, síntomas o persistencia— sigue pendiente de confirmación.

ERA5 es retrospectivo y su disponibilidad operacional no está acreditada. Las ubicaciones se repiten, algunas comparten celdas y la identidad física de Jupiá conserva incertidumbre. La cobertura por ubicación y campaña es desigual. H0 y h7 usan poblaciones distintas. Las asociaciones y las diferencias observadas no demuestran causalidad ni generalización a nuevas regiones.

En este diseño los modelos climáticos seleccionados muestran recall limitado. Para +7, persistencia supera al modelo climático seleccionado en la evaluación histórica. El trabajo conserva su carácter experimental y no respalda una alerta agronómica validada. Las próximas investigaciones requerirían fechas de disponibilidad verificadas y una evaluación prospectiva inédita definida antes de analizar resultados.

## Reproducibilidad y verificación

"""
    text += f"Verificamos {verification['metricas_recalculadas']} registros de métricas, {verification['predicciones_verificadas']} predicciones y ocho modelos recargados. Error máximo de puntuaciones recargadas: {verification['error_maximo_modelos']:.3g}. Dos ejecuciones completas coincidieron en selección, métricas y probabilidades con tolerancia absoluta 10⁻¹². La selección congelada precede a la evaluación histórica y mantiene el mismo hash.\n\n"
    text += """Los hashes de entradas, configuración, código y selección se conservan en los manifiestos. entorno.json identifica versiones y commit base; los scripts de esta etapa todavía no confirmados se identifican por SHA-256, para evitar atribuirlos al commit anterior.

Los modelos y predicciones por muestra permanecen en results y models, excluido de Git. El reporte agregado conserva las cifras y figuras revisables. Para reproducir el experimento se necesitan los insumos locales autorizados. Las etiquetas Epagri tienen condiciones de redistribución pendientes de aclaración. Datos meteorológicos de Open-Meteo/ERA5, CC BY 4.0, agregados por el equipo.

La ejecución utiliza scripts/entrenar.py y scripts/evaluar.py, scripts/verificar.py y scripts/generar_informe.py, con config/proyecto.json. Los notebooks finales, presentación y portal corresponden a una etapa posterior.
"""
    (destination/'INFORME.md').write_text(text, encoding='utf-8')
    write_json(destination/'manifiesto_reporte.json', {
        'fuente_seleccion_sha256': sha(source/'seleccion.json'),
        'script_sha256': sha(__file__), 'salidas': {
            str(p.relative_to(destination)): sha(p) for p in sorted(destination.rglob('*')) if p.is_file()}})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--destination', type=Path, required=True)
    args = parser.parse_args()
    report(args.source, args.destination)
