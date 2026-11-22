"""Auditoría de recepción y propuesta temporal. No importa ni ejecuta entrenamientos."""
from pathlib import Path
from zipfile import ZipFile
from collections import Counter
import argparse
import hashlib
import json
import shutil
import numpy as np
import pandas as pd
from sklearn.metrics import (f1_score, precision_score, recall_score, accuracy_score,
                             balanced_accuracy_score, roc_auc_score, average_precision_score)

ROOT = Path(__file__).resolve().parents[1]
FEATURES = ['hr_media_pct', 'hr_min_pct', 'hr_max_pct', 'temperatura_media_c',
            'temperatura_min_c', 'temperatura_max_c', 'precipitacion_mm',
            'viento_vel_media_10m_ms', 'viento_u_media_10m_ms', 'viento_v_media_10m_ms']


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def json_write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str,
                               allow_nan=False), encoding='utf-8')


def describe(group, label):
    if group.empty:
        return {'n': 0}
    start, end = group.ref.min(), group.ref.max()
    return dict(n=len(group), negativos=int(group[label].eq(0).sum()),
                positivos=int(group[label].eq(1).sum()),
                ubicaciones=len(group[['latitud', 'longitud']].drop_duplicates()),
                punto_ids=group.punto_id.nunique(),
                ref_inicio=str(start.date()), ref_fin=str(end.date()),
                duracion_dias=int((end-start).days), dias_inclusivos=int((end-start).days)+1,
                fechas_observadas=group.ref.nunique(),
                etiqueta_inicio=str(group.target.min().date()),
                etiqueta_fin=str(group.target.max().date()),
                meteo_inicio=str((start-pd.Timedelta(days=14)).date()),
                meteo_fin=str((end-pd.Timedelta(days=1)).date()))


def run(args):
    local = ROOT/'data/raw/etapa_b_20261122'
    report = ROOT/'reports/etapa_b'
    local.mkdir(parents=True, exist_ok=True)
    report.mkdir(parents=True, exist_ok=True)
    manifest = {'etapa': 'B', 'recibido': '2026-11-22', 'entrenamientos_ejecutados': False,
                'entradas': [], 'disponibilidad_operativa': 'no acreditada'}
    for source in [args.dataset_zip, args.results_zip]:
        source = source.resolve()
        copied = local/source.name
        if copied.exists():
            assert sha(copied) == sha(source), 'Recepción existente distinta'
        else:
            shutil.copy2(source, copied)
        entry = {'origen': str(source), 'copia': str(copied.relative_to(ROOT)),
                 'sha256': sha(source), 'bytes': source.stat().st_size, 'miembros': []}
        folder = local/source.stem
        folder.mkdir(exist_ok=True)
        with ZipFile(copied) as archive:
            assert archive.testzip() is None
            for member in archive.infolist():
                destination = (folder/member.filename).resolve()
                assert destination.is_relative_to(folder.resolve()), 'Ruta ZIP insegura'
                if member.is_dir():
                    continue
                content = archive.read(member)
                destination.parent.mkdir(parents=True, exist_ok=True)
                if destination.exists():
                    assert destination.read_bytes() == content
                else:
                    destination.write_bytes(content)
                entry['miembros'].append({'archivo': member.filename, 'bytes': len(content),
                                           'sha256': hashlib.sha256(content).hexdigest()})
        manifest['entradas'].append(entry)
    # Originales de respaldo se copian para que la auditoría sea repetible sin rutas externas.
    backing = local/'respaldo'
    backing.mkdir(exist_ok=True)
    sources = [
        args.original_root/'data/raw/monitoreo_deduplicado.csv',
        args.original_root/'data/raw/monitoreo_deduplicado_filtrado_nasa.csv',
        args.original_root/'data/processed/features_7dias.csv',
        args.original_root/'data/processed/features_14dias.csv',
        args.original_root/'data/processed/monitoreo_ml_14dias.csv']
    manifest['respaldo'] = []
    for source in sources:
        copied = backing/source.name
        if copied.exists():
            assert sha(copied) == sha(source)
        else:
            shutil.copy2(source, copied)
        manifest['respaldo'].append({'origen': str(source.resolve()),
                                     'archivo': str(copied.relative_to(ROOT)), 'sha256': sha(source)})
    dataset = local/args.dataset_zip.stem
    preliminary = local/args.results_zip.stem
    original = pd.read_csv(backing/'monitoreo_deduplicado.csv')
    data = pd.read_csv(dataset/'datos_horizontes_0_y_7.csv')
    future = pd.read_csv(dataset/'evaluaciones_futuras_validas.csv')
    negative = pd.read_csv(dataset/'transiciones_desde_negativo.csv')
    assert data.id_medicion.is_unique and not data.duplicated(['latitud','longitud','fecha_objetivo']).any()
    assert not original.duplicated(['latitud','longitud','fecha_registro']).any()
    assert original.conflicto_clase_mismo_punto_fecha.eq(0).all()
    assert not any('iith' in c.lower() for c in data)
    expected_features = [f'{feature}_{w}d' for w in [7,14] for feature in FEATURES]
    assert np.isfinite(data[expected_features].to_numpy()).all()
    assert data.resultado_dia_0.isin(['Positivo','Negativo']).all()
    assert data.objetivo_dia_0.eq(data.resultado_dia_0.eq('Positivo').astype(int)).all()
    data['ref'] = pd.to_datetime(data.fecha_objetivo)
    data['target'] = pd.to_datetime(data.fecha_dia_mas_7)
    assert (data.target-data.ref).dt.days.eq(7).all()
    lookup = original.set_index(['latitud','longitud','fecha_registro']).clase_original.to_dict()
    exclusions = []
    for row in data.itertuples():
        assert lookup[(row.latitud,row.longitud,row.fecha_objetivo)] == row.resultado_dia_0
        actual = lookup.get((row.latitud,row.longitud,row.fecha_dia_mas_7))
        if actual in ['Positivo','Negativo']:
            assert row.resultado_dia_mas_7 == actual
            assert row.objetivo_dia_mas_7 == int(actual == 'Positivo')
        else:
            assert pd.isna(row.objetivo_dia_mas_7)
            exclusions.append({'id_medicion': row.id_medicion, 'fecha_ref': row.fecha_objetivo,
                               'fecha_esperada': row.fecha_dia_mas_7,
                               'motivo': 'no_evaluado' if actual is not None else 'sin_registro'})
    for file in ['datos_horizontes_0_y_7.csv','evaluaciones_futuras_validas.csv',
                 'transiciones_desde_negativo.csv','LEEME.md']:
        assert sha(dataset/file) == sha(preliminary/'dataset_royasoja_sin_iith'/file)
    assert set(future.id_medicion) == set(data.loc[data.objetivo_dia_mas_7.notna(),'id_medicion'])
    assert set(negative.id_medicion) == set(future.loc[future.objetivo_dia_0.eq(0),'id_medicion'])
    for supplied in [future, negative]:
        cols=supplied.columns.tolist()
        pd.testing.assert_frame_equal(supplied.sort_values('id_medicion').reset_index(drop=True),
            data.set_index('id_medicion').loc[supplied.id_medicion].reset_index()[cols]
                .sort_values('id_medicion').reset_index(drop=True),check_dtype=False)
    max_diffs = {}
    for w in [7,14]:
        previous = pd.read_csv(backing/f'features_{w}dias.csv').set_index('id_medicion')
        for col in FEATURES:
            delta = np.abs(data[f'{col}_{w}d'].to_numpy()-previous.loc[data.id_medicion,col].to_numpy())
            max_diffs[f'{col}_{w}d'] = float(delta.max())
        assert max(max_diffs.values()) < 1e-9
    daily = pd.read_csv(backing/'monitoreo_ml_14dias.csv')
    assert daily.groupby('id_medicion').size().eq(14).all()
    assert set(data.id_medicion)==set(daily.id_medicion)
    assert (pd.to_datetime(daily.fecha_objetivo)-pd.to_datetime(daily.fecha_dato)).dt.days.eq(daily.dias_anteriores).all()
    for w in [7,14]:
        aggregation={c: ('min' if '_min_' in c else 'max' if '_max_' in c
                         else 'sum' if c=='precipitacion_mm' else 'mean') for c in FEATURES}
        rebuilt=daily[daily.dias_anteriores<=w].groupby('id_medicion').agg(aggregation)
        for c in FEATURES:
            assert np.allclose(rebuilt.loc[data.id_medicion,c],data[f'{c}_{w}d'],atol=1e-10)
        assert (data[f'hr_min_pct_{w}d']<=data[f'hr_media_pct_{w}d']).all()
        assert (data[f'hr_media_pct_{w}d']<=data[f'hr_max_pct_{w}d']).all()
        assert data[[f'hr_{s}_pct_{w}d' for s in ['media','min','max']]].apply(lambda x:x.between(0,100)).all().all()
        assert (data[f'temperatura_min_c_{w}d']<=data[f'temperatura_media_c_{w}d']).all()
        assert (data[f'temperatura_media_c_{w}d']<=data[f'temperatura_max_c_{w}d']).all()
        assert data[f'precipitacion_mm_{w}d'].ge(0).all()
        assert data[f'viento_vel_media_10m_ms_{w}d'].ge(0).all()
        assert (np.hypot(data[f'viento_u_media_10m_ms_{w}d'],data[f'viento_v_media_10m_ms_{w}d'])<=data[f'viento_vel_media_10m_ms_{w}d']+1e-10).all()
    coordinate_future=0
    for row in data[data.objetivo_dia_mas_7.notna()].itertuples():
        matched=original[(original.latitud==row.latitud)&(original.longitud==row.longitud)&
                         (original.fecha_registro==row.fecha_dia_mas_7)]
        coordinate_future += int(matched.iloc[0].punto_id!=row.punto_id)
    # Dependencias originales completas de disponibilidad horaria: hashes y cobertura.
    hourly_evidence=[]
    for folder in ['weather','weather_previos','weather_14dias']:
        for path in sorted((args.original_root/'data/raw'/folder).glob('[0-9]*.json')):
            payload=json.loads(path.read_text(encoding='utf-8'))
            h=payload['hourly']
            assert len(h['time'])==len(set(h['time']))
            assert payload['utc_offset_seconds']==-10800
            assert payload['hourly_units']['wind_speed_10m']=='m/s'
            assert all(len(h[v])==len(h['time']) and all(x is not None for x in h[v])
                       for v in ['temperature_2m','relative_humidity_2m','precipitation','rain','wind_speed_10m','wind_direction_10m'])
            hourly_evidence.append({'archivo':f'{folder}/{path.name}','sha256':sha(path),
                                    'horas':len(h['time']),'inicio':min(h['time']),'fin':max(h['time'])})
    json_write(report/'respaldo_horario_hashes.json',hourly_evidence)
    summary = dict(original_n=len(original), original_clases=original.clase_original.value_counts().to_dict(),
                   n_h0=len(data), n_h7=len(future), n_negativos_h7=len(negative),
                   clases_h0=data.objetivo_dia_0.value_counts().to_dict(),
                   clases_h7=future.objetivo_dia_mas_7.value_counts().to_dict(),
                   ubicaciones=len(data[['latitud','longitud']].drop_duplicates()),
                   punto_ids=data.punto_id.nunique(), exclusiones_h7=dict(Counter(e['motivo'] for e in exclusions)),
                   fecha_igual_primera_consulta=int(original.fecha_registro.eq(original.primera_consulta).sum()),
                   n_snapshots_min=int(original.n_snapshots.min()), n_snapshots_max=int(original.n_snapshots.max()),
                   diferencia_maxima_features=max(max_diffs.values()), variables=expected_features,
                   fecha_ref_min=data.fecha_objetivo.min(),fecha_ref_max=data.fecha_objetivo.max())
    summary['pares_h7_entre_punto_ids_distintos']=coordinate_future
    summary['respuestas_horarias_auditadas']=len(hourly_evidence)
    pd.DataFrame(exclusions).to_csv(local/'exclusiones_h7.csv',index=False)
    pd.DataFrame({'variable':list(max_diffs),'diferencia_maxima':list(max_diffs.values())}).to_csv(report/'comparacion_features.csv',index=False)
    folds = []; splits = {}; partitions = []
    for horizon in [0,7]:
        label = f'objetivo_dia_{0 if horizon==0 else "mas_7"}'
        population = data[data[label].notna()].copy().sort_values(['ref','id_medicion'])
        population['target'] = population.ref+pd.Timedelta(days=horizon)
        population.to_csv(local/f'poblacion_h{horizon}_auditada.csv',index=False)
        test = population[population.target.between('2026-01-01','2026-03-31')]
        v = test.ref.min()
        train = population[population.target < v-pd.Timedelta(days=14)]
        purged = population[(population.target < pd.Timestamp('2026-01-01')) & ~population.index.isin(train.index)]
        assert train[label].nunique()==test[label].nunique()==2
        splits[str(horizon)] = dict(evaluacion=describe(test,label), desarrollo=describe(train,label),
                                    purga=describe(purged,label), inicio_ref_evaluacion=str(v.date()),
                                    regla='fecha_objetivo_train < inicio_ref_validacion - 14 dias')
        date_list = sorted(train.ref.unique())
        assert len(date_list)>21
        for fold in range(3):
            valid_dates=date_list[len(date_list)-21+fold*7:len(date_list)-14+fold*7]
            valid = train[train.ref.isin(valid_dates)]
            start=valid.ref.min()
            fit=train[train.target < start-pd.Timedelta(days=14)]
            purga=train[(train.ref<start)&~train.index.isin(fit.index)]
            assert fit[label].nunique()==valid[label].nunique()==2
            assert fit.target.max()<start-pd.Timedelta(days=14)
            for role, subset in [('entrenamiento',fit),('validacion',valid),('purga',purga)]:
                folds.append({'horizonte':horizon,'fold':fold+1,'rol':role,**describe(subset,label)})
            partitions.append({'horizonte':horizon,'fold':fold+1,'train_ids':fit.id_medicion.tolist(),
                               'validation_ids':valid.id_medicion.tolist(),'purged_ids':purga.id_medicion.tolist()})
        partitions.append({'horizonte':horizon,'fold':'historico','train_ids':train.id_medicion.tolist(),
                           'test_ids':test.id_medicion.tolist(),'purged_ids':purged.id_medicion.tolist()})
        for w in [7,14]:
            descriptive=train[[f'{c}_{w}d' for c in FEATURES]].describe()
            descriptive.to_csv(report/f'eda_desarrollo_h{horizon}_{w}d.csv')
            train[[f'{c}_{w}d' for c in FEATURES]].corr().to_csv(report/f'correlaciones_desarrollo_h{horizon}_{w}d.csv')
            extrema=[]
            for c in FEATURES:
                values=train[f'{c}_{w}d']
                q1,q3=values.quantile([.25,.75])
                extrema.append({'variable':c,'ventana':w,'horizonte':horizon,
                                'extremos_iqr':int(((values<q1-1.5*(q3-q1))|(values>q3+1.5*(q3-q1))).sum()),
                                'tratamiento':'conservar; descriptor exploratorio'})
            pd.DataFrame(extrema).to_csv(report/f'extremos_desarrollo_h{horizon}_{w}d.csv',index=False)
        train.assign(mes=train.ref.dt.strftime('%Y-%m')).groupby(['mes',label]).size().unstack(fill_value=0).to_csv(report/f'cobertura_mensual_desarrollo_h{horizon}.csv')
        pd.crosstab(train.resultado_dia_0,train[label]).to_csv(report/f'eda_clases_h{horizon}.csv')
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, axes=plt.subplots(1,2,figsize=(11,4))
        train[label].value_counts().reindex([0,1],fill_value=0).plot.bar(ax=axes[0],color=['#19685b','#c77937'])
        axes[0].set(xticklabels=['Negativo','Positivo'],ylabel='Observaciones',title=f'Desarrollo h{horizon}, n={len(train)}')
        monthly=train.assign(mes=train.ref.dt.strftime('%Y-%m')).groupby(['mes',label]).size().unstack(fill_value=0)
        monthly.plot.bar(stacked=True,ax=axes[1],color=['#19685b','#c77937'])
        axes[1].set(title='Cobertura por mes de referencia',ylabel='Observaciones',xlabel='Mes')
        fig.tight_layout()
        fig.savefig(report/f'eda_desarrollo_h{horizon}.png',dpi=140)
        plt.close(fig)
    # Solo recálculo de métricas recibidas. El script de entrenamiento del ZIP nunca se importa.
    experiment = preliminary/'experimentos_royasoja_sin_iith'
    comparison=pd.read_csv(experiment/'comparacion_modelos.csv')
    predictions=pd.read_csv(experiment/'predicciones_prueba.csv')
    metric_checks=[]
    for row in comparison.to_dict('records'):
        pred=predictions[predictions.config.eq(row['config'])].copy()
        assert pred.id_medicion.is_unique and len(pred)==row['n_prueba']
        y=pred.real; prob=pred.probabilidad; hard=pred.prediccion
        assert np.isfinite(prob).all() and prob.between(0,1).all()
        assert hard.eq(prob.ge(.5).astype(int)).all()
        source=data.set_index('id_medicion').loc[pred.id_medicion]
        label='objetivo_dia_0' if row['horizonte']==0 else 'objetivo_dia_mas_7'
        assert np.array_equal(y.to_numpy(),source[label].to_numpy())
        expected_date=source.ref+pd.Timedelta(days=int(row['horizonte']))
        assert np.array_equal(pd.to_datetime(pred.fecha_resultado).to_numpy(),expected_date.to_numpy())
        expected_ids=data.loc[(data[label].notna()) & ((data.ref+pd.Timedelta(days=int(row['horizonte'])))>=pd.Timestamp('2026-01-01')),'id_medicion']
        assert set(pred.id_medicion)==set(expected_ids)
        vals=dict(f1=f1_score(y,hard),precision=precision_score(y,hard,zero_division=0),
                  recall=recall_score(y,hard),accuracy=accuracy_score(y,hard),
                  balanced_accuracy=balanced_accuracy_score(y,hard),roc_auc=roc_auc_score(y,prob),
                  average_precision=average_precision_score(y,prob))
        for key,value in vals.items():
            assert np.isclose(value,row['test_'+key],atol=1e-12)
        metric_checks.append({'config':row['config'],'n':len(pred),'metricas_coinciden':True})
    pd.DataFrame(metric_checks).to_csv(report/'verificacion_metricas_antecedentes.csv',index=False)
    # Manifest local includes identifiable locations; public receipt contains only file hashes.
    json_write(local/'manifiesto_local.json',manifest)
    json_write(local/'particiones_propuestas_ids.json',partitions)
    pd.DataFrame(folds).to_csv(report/'folds_propuestos.csv',index=False)
    json_write(report/'poblaciones_y_cortes.json',splits)
    json_write(report/'auditoria.json',summary)
    public_manifest={'recibido':manifest['recibido'],'entrenamientos_ejecutados':False,
                     'entradas':[{k:v for k,v in e.items() if k!='origen'} for e in manifest['entradas']],
                     'respaldo':[{k:v for k,v in e.items() if k!='origen'} for e in manifest['respaldo']]}
    json_write(report/'procedencia_y_hashes.json',public_manifest)
    print(json.dumps({'auditoria':summary,'cortes':splits,'folds':folds},ensure_ascii=False,default=str,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--dataset-zip',type=Path,required=True)
    parser.add_argument('--results-zip',type=Path,required=True)
    parser.add_argument('--original-root',type=Path,required=True)
    run(parser.parse_args())
