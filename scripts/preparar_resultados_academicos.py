"""Exporta curvas agregadas desde predicciones oficiales; no ajusta modelos."""
import json
import pandas as pd
from sklearn.metrics import roc_curve, precision_recall_curve, confusion_matrix
from procesar_datos import ROOT, configured, sha, write_json, require


def run():
    output = configured('resultados')/'academicos'
    output.mkdir(exist_ok=True)
    source = configured('resultados')/'predictions/predicciones.csv'
    predictions = pd.read_csv(source)
    comparison = pd.read_csv(configured('resultados')/'comparacion.csv').set_index('config')
    curves = []
    for name, group in predictions[predictions.rol.eq('historico')].groupby('config'):
        cm = confusion_matrix(group.y, group.prediccion, labels=[0, 1]).ravel()
        require(list(cm) == comparison.loc[name, ['historico_tn','historico_fp','historico_fn','historico_tp']].tolist(), 'Matriz distinta')
        fpr,tpr,_ = roc_curve(group.y, group.probabilidad)
        precision,recall,_ = precision_recall_curve(group.y, group.probabilidad)
        curves.append(dict(config=name, horizonte=int(group.horizonte.iloc[0]),
            fpr=fpr.tolist(),tpr=tpr.tolist(),precision=precision.tolist(),recall=recall.tolist()))
    write_json(output/'curvas.json',curves)
    summary = configured('datos')/'resumen_particiones.csv'
    pd.read_csv(summary).to_csv(output/'particiones.csv',index=False)
    write_json(output/'manifiesto.json',dict(
        origen_predicciones_sha256=sha(source),origen_particiones_sha256=sha(summary),
        seleccion_sha256=sha(configured('resultados')/'seleccion.json'),
        script_sha256=sha(__file__),contenido='Coordenadas de curvas agregadas; sin IDs, fechas individuales ni puntuaciones por muestra',
        salidas={p.name:sha(p) for p in [output/'curvas.json',output/'particiones.csv']}))


if __name__=='__main__':
    run()
