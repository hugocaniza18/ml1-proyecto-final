"""Controles de contrato, fuga temporal y métricas independientes del entrenamiento."""
import json
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import f1_score, confusion_matrix
from data_pipeline import ROOT, PROCESSED, REPORTS, FEATURES, save_json

def verify():
    long=pd.read_csv(PROCESSED/'monitoreo_ml_14dias.csv')
    short=pd.read_csv(PROCESSED/'monitoreo_ml_7dias.csv')
    a=pd.read_csv(PROCESSED/'features_7dias.csv').set_index('id_medicion')
    b=pd.read_csv(PROCESSED/'features_14dias.csv').set_index('id_medicion')
    s=json.loads((PROCESSED/'particiones.json').read_text())
    assert len(long)==5992 and len(short)==2996 and len(a)==len(b)==428
    assert a.index.equals(b.index)
    pd.testing.assert_frame_equal(short.reset_index(drop=True),long[long.dias_anteriores<=7].reset_index(drop=True))
    assert not long.duplicated(['id_medicion','dias_anteriores']).any()
    assert long.groupby('id_medicion').dias_anteriores.apply(lambda x:set(x)==set(range(1,15))).all()
    assert ((pd.to_datetime(long.fecha_objetivo)-pd.to_datetime(long.fecha_dato)).dt.days==long.dias_anteriores).all()
    assert long.iith_h.between(0,24).all()
    assert long[long.estado_registro_roya!='evaluado'].roya_anterior.isna().all()
    for window,table in [(7,a),(14,b)]:
        expected=long[long.dias_anteriores<=window].groupby('id_medicion').iith_h.sum()
        assert table.iith_h.equals(expected.reindex(table.index))
        assert np.isfinite(table[FEATURES+['iith_h']].to_numpy()).all()
    partitions=[set(s[k]) for k in ['train_ids','test_ids','purged_ids']]
    assert not partitions[0]&partitions[1] and not partitions[0]&partitions[2] and not partitions[1]&partitions[2]
    assert set.union(*partitions)==set(a.index)
    for fold in s['folds']:
        tr=a.loc[fold['train_ids']];va=a.loc[fold['validation_ids']]
        assert len(va.fecha_objetivo.unique())==7
        assert tr.objetivo_positivo.nunique()==va.objetivo_positivo.nunique()==2
        assert pd.to_datetime(tr.fecha_objetivo).max()<pd.to_datetime(va.fecha_objetivo).min()-pd.Timedelta(days=14)
        assert not set(tr.index)&set(va.index)
        assert set(tr.index)|set(va.index)<=partitions[0]
    assert pd.to_datetime(a.loc[s['train_ids']].fecha_objetivo).max()<pd.Timestamp(s['test_start'])-pd.Timedelta(days=14)
    results=json.loads((REPORTS/'resultados.json').read_text())
    frozen=json.loads((REPORTS/'seleccion_congelada.json').read_text())
    assert results['selected_config']==frozen['selected']['config']
    predictions=pd.read_csv(REPORTS/'predicciones_prueba.csv')
    assert len(predictions)==8*66
    for r in results['results']:
        p=predictions[predictions.config==r['config']].set_index('id_medicion')
        assert set(p.index)==partitions[1]
        assert np.isclose(f1_score(p.y_true,p.prediccion),r['test_f1'])
        model=joblib.load(ROOT/'models'/f"{r['config']}.joblib")
        table=a if r['dias']==7 else b
        actual=model['pipeline'].predict_proba(table.loc[p.index,model['features']])[:,1]
        assert np.allclose(actual,p.probabilidad_positivo)
    selected=predictions[predictions.config==results['selected_config']]
    report={'passed':True,'checks':['428 objetivos únicos','ventanas completas y coincidentes','día objetivo excluido','ausencia distinta de negativo','purga temporal en folds y prueba','métricas recalculadas','modelos guardados reproducen probabilidades'],'selected_confusion_counts':confusion_matrix(selected.y_true,selected.prediccion).tolist()}
    save_json(REPORTS/'verificacion.json',report)
    print(json.dumps(report,ensure_ascii=False,indent=2))

if __name__=='__main__':verify()
