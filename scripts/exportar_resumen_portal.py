"""Exporta exclusivamente agregados oficiales para el portal visual."""
import json
import pandas as pd
from procesar_datos import ROOT, FEATURES


def run():
    results = ROOT/'results'
    selected = json.loads((results/'seleccion.json').read_text(encoding='utf-8'))['selected']
    comparison = pd.read_csv(results/'comparacion.csv')
    models = []
    for row in comparison.to_dict('records'):
        models.append({
            'id': row['config'], 'horizon': row['horizonte'], 'window': row['ventana'],
            'selected': selected[str(row['horizonte'])] == row['config'],
            'cv': row['cv_f1'], 'f1': row['historico_f1'],
            'precision': row['historico_precision'], 'recall': row['historico_recall'],
            'n': int(row['historico_n']),
            'matrix': [[int(row['historico_tn']), int(row['historico_fp'])],
                       [int(row['historico_fn']), int(row['historico_tp'])]],
        })
    classes = []
    for h in [0, 7]:
        counts = pd.read_csv(results/f'eda/eda_clases_h{h}.csv').iloc[:, 1:].sum()
        classes.append({'horizon': h, 'negative': int(counts.iloc[0]), 'positive': int(counts.iloc[1])})
    assert [c['negative']+c['positive'] for c in classes] == [359, 313]
    monthly = pd.read_csv(results/'eda/cobertura_mensual_desarrollo_h0.csv').set_index('mes')
    # Los meses sin registros permanecen como ausencia de observaciones.
    months = pd.period_range(monthly.index.min(), monthly.index.max(), freq='M')
    coverage = [{'month': str(m), 'negative': int(monthly.loc[str(m)].iloc[0]) if str(m) in monthly.index else None,
                 'positive': int(monthly.loc[str(m)].iloc[1]) if str(m) in monthly.index else None} for m in months]
    assert sum((x['negative'] or 0)+(x['positive'] or 0) for x in coverage) == 359
    variables = [FEATURES[i] for i in [0, 3, 6, 7, 8, 9]]
    corr = pd.read_csv(results/'eda/correlaciones_desarrollo_h0_14d.csv', index_col=0)
    keys = [v+'_14d' for v in variables]
    payload = dict(models=models, classes=classes, coverage=coverage,
                   correlations=corr.loc[keys, keys].values.tolist())
    (ROOT/'web/app/portal-data.json').write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')


if __name__ == '__main__':
    run()
