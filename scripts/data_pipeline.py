"""Audita fuentes y construye ventanas retrospectivas. Sin imputar ni repetir objetivos."""
from pathlib import Path
import argparse, hashlib, json, time
from urllib.parse import urlencode
from urllib.request import urlopen
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW, PROCESSED, REPORTS = ROOT/'data/raw', ROOT/'data/processed', ROOT/'reports'
VARIABLES = ['temperature_2m','relative_humidity_2m','precipitation','rain','wind_speed_10m','wind_direction_10m']
FEATURES = ['hr_media_pct','hr_min_pct','hr_max_pct','temperatura_media_c','temperatura_min_c','temperatura_max_c','precipitacion_mm','viento_vel_media_10m_ms','viento_u_media_10m_ms','viento_v_media_10m_ms']

def save_json(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')

def audit():
    original = pd.read_csv(RAW/'monitoreo_deduplicado.csv')
    source = pd.read_csv(RAW/'monitoreo_deduplicado_filtrado_nasa.csv')
    old = pd.read_csv(RAW/'monitoreo_ml_7dias.csv')
    keys = ['latitud','longitud','fecha']
    assert len(source)==428 and len(original)==650
    assert not source.duplicated(keys).any()
    assert not original.duplicated(['latitud','longitud','fecha_registro']).any()
    assert (original.conflicto_clase_mismo_punto_fecha==0).all()
    expected = original[original.clase_original.isin(['Positivo','Negativo'])].rename(columns={'fecha_registro':'fecha','clase_original':'resultado'})
    compared = source.merge(expected, on=keys, validate='one_to_one', suffixes=('', '_original'))
    assert len(compared)==428 and (compared.resultado==compared.resultado_original).all()
    targets = old[['id_medicion','latitud','longitud','fecha_objetivo','resultado_objetivo']].drop_duplicates()
    assert len(targets)==428 and targets.id_medicion.is_unique
    checked=targets.rename(columns={'fecha_objetivo':'fecha','resultado_objetivo':'resultado'}).merge(source,on=keys+['resultado'],validate='one_to_one')
    assert len(checked)==428
    assert old.groupby('id_medicion').size().eq(7).all()
    assert set(original.clase_codigo)=={79,80,82}
    assert original.groupby('clase_codigo').clase_original.unique().apply(list).to_dict()=={79:['Positivo'],80:['Negativo'],82:['Não Avaliado']}
    targets=targets.merge(original[['latitud','longitud','fecha_registro','municipio','punto_id']],left_on=['latitud','longitud','fecha_objetivo'],right_on=['latitud','longitud','fecha_registro'],validate='one_to_one').drop(columns='fecha_registro')
    targets['objetivo_positivo']=targets.resultado_objetivo.eq('Positivo').astype(int)
    targets.to_csv(PROCESSED/'objetivos.csv',index=False)
    report={'mediciones':len(targets),'registros_originales':len(original),'clases':source.resultado.value_counts().to_dict(),'coordenadas':len(source[['latitud','longitud']].drop_duplicates()),'fechas':source.fecha.nunique(),'fecha_min':source.fecha.min(),'fecha_max':source.fecha.max(),'ids_unicos':True,'correspondencia_original':True,'original_sha256':hashlib.sha256((RAW/'monitoreo_deduplicado.csv').read_bytes()).hexdigest()}
    save_json(REPORTS/'auditoria.json',report)
    return targets,original

def load_hourly(entry, folder):
    p=RAW/folder/f"{entry['id']:02d}.json"
    payload=json.loads(p.read_text(encoding='utf-8'))
    assert payload['utc_offset_seconds']==-10800
    assert payload['hourly_units']['wind_speed_10m']=='m/s'
    frame=pd.DataFrame(payload['hourly']).set_index('time')
    assert frame.index.is_unique
    return frame, payload

def build(download=False):
    targets, original = audit()
    manifest=json.loads((RAW/'weather/manifest.json').read_text())
    extension=RAW/'weather_14dias';extension.mkdir(exist_ok=True)
    sources=[]; long=[]; provenance=[]; missing=[]
    historical={(r.latitud,r.longitud,r.fecha_registro):r.clase_original for r in original.itertuples()}
    extras=[]
    for entry in manifest['requests']:
        lat,lon=float(entry['latitude']),float(entry['longitude'])
        selected=targets[(targets.latitud==lat)&(targets.longitud==lon)]
        frames=[]
        for folder in ['weather','weather_previos']:
            f,payload=load_hourly(entry,folder);frames.append(f)
            m=json.loads((RAW/folder/'manifest.json').read_text())
            e=next(x for x in m['requests'] if x['id']==entry['id'])
            sources.append({'id':entry['id'],'folder':folder,'latitude':lat,'longitude':lon,'url':e['url'],'sha256':hashlib.sha256((RAW/folder/f"{entry['id']:02d}.json").read_bytes()).hexdigest()})
        hourly=pd.concat(frames)
        assert hourly.index.is_unique
        needed={str((pd.Timestamp(d)-pd.Timedelta(days=lag)).date()) for d in selected.fecha_objetivo for lag in range(1,15)}
        absent=sorted(needed-set(hourly.index.str[:10]))
        if absent:
            params=dict(latitude=lat,longitude=lon,start_date=absent[0],end_date=absent[-1],hourly=','.join(VARIABLES),models='era5',timezone='America/Sao_Paulo',wind_speed_unit='ms')
            url='https://archive-api.open-meteo.com/v1/archive?'+urlencode(params)
            path=extension/f"{entry['id']:02d}.json"
            if not path.exists() and download:
                for attempt in range(4):
                    try:
                        with urlopen(url,timeout=45) as resp: data=json.load(resp)
                        assert 'hourly' in data
                        save_json(path,data);break
                    except Exception:
                        if attempt==3: raise
                        time.sleep(3*(attempt+1))
                time.sleep(.5)
            if path.exists():
                f,extra=load_hourly(entry,'weather_14dias')
                assert all(extra[k]==payload[k] for k in ['latitude','longitude','elevation'])
                hourly=pd.concat([hourly,f]);assert hourly.index.is_unique
                extra_entry={'id':entry['id'],'folder':'weather_14dias','latitude':lat,'longitude':lon,'url':url,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
                extras.append(extra_entry);sources.append(extra_entry)
        for r in selected.itertuples():
            daily=[]
            for lag in range(1,15):
                day=str((pd.Timestamp(r.fecha_objetivo)-pd.Timedelta(days=lag)).date())
                stamps=[day+f'T{h:02}:00' for h in range(24)]
                if not set(stamps)<=set(hourly.index):
                    missing.append({'id_medicion':r.id_medicion,'fecha':day,'motivo':'horas_ausentes'});continue
                h=hourly.loc[stamps,VARIABLES]
                if not np.isfinite(h.to_numpy()).all():
                    missing.append({'id_medicion':r.id_medicion,'fecha':day,'motivo':'valor_ausente'});continue
                t,rh,pr,rain,speed,direction=[h[v].to_numpy() for v in VARIABLES]
                assert ((rh>=0)&(rh<=100)).all() and (pr>=0).all() and (rain>=0).all() and (speed>=0).all()
                u=float(-np.mean(speed*np.sin(np.radians(direction))));v=float(-np.mean(speed*np.cos(np.radians(direction))))
                hist=historical.get((lat,lon,day))
                daily.append(dict(id_medicion=r.id_medicion,latitud=lat,longitud=lon,fecha_objetivo=r.fecha_objetivo,resultado_objetivo=r.resultado_objetivo,dias_anteriores=lag,fecha_dato=day,roya_anterior=hist if hist in ['Positivo','Negativo'] else '',estado_registro_roya='evaluado' if hist in ['Positivo','Negativo'] else ('no_evaluado' if hist is not None else 'sin_registro'),hr_media_pct=float(rh.mean()),hr_min_pct=float(rh.min()),hr_max_pct=float(rh.max()),temperatura_min_c=float(t.min()),temperatura_max_c=float(t.max()),temperatura_media_c=float(t.mean()),precipitacion_mm=float(pr.sum()),lluvia_mm=float(rain.sum()),viento_vel_media_10m_ms=float(speed.mean()),viento_vel_max_10m_ms=float(speed.max()),viento_dir_10m_grados=float(np.degrees(np.arctan2(-u,-v))%360) if np.hypot(u,v)>1e-8 else None,viento_u_media_10m_ms=u,viento_v_media_10m_ms=v,iith_h=int(((rh>=85)&(t>=15)&(t<=28)).sum()),horas_hr_ge90_h=int((rh>=90).sum())))
            if len(daily)==14: long.extend(daily)
            provenance.append(dict(id_medicion=r.id_medicion,om_latitud_celda=payload['latitude'],om_longitud_celda=payload['longitude'],om_elevacion_m=payload['elevation'],fuente='Open-Meteo ERA5',zona_horaria='America/Sao_Paulo',horas_por_dia=24,hmf_estado='pendiente',ciram_meteorologia_estado='pendiente'))
        print(f"Coordenada {entry['id']}/22: {len(selected)} mediciones",flush=True)
    save_json(extension/'manifest.json',{'requests':extras})
    save_json(PROCESSED/'fuentes.json',sources)
    frame=pd.DataFrame(long).sort_values(['id_medicion','dias_anteriores'])
    frame.to_csv(PROCESSED/'monitoreo_ml_14dias.csv',index=False)
    frame[frame.dias_anteriores<=7].to_csv(PROCESSED/'monitoreo_ml_7dias.csv',index=False)
    pd.DataFrame(provenance).to_csv(PROCESSED/'procedencia.csv',index=False)
    save_json(REPORTS/'exclusiones.json',missing)
    accepted=targets[targets.id_medicion.isin(frame.id_medicion)].copy()
    aggregations={'hr_media_pct':'mean','hr_min_pct':'min','hr_max_pct':'max','temperatura_media_c':'mean','temperatura_min_c':'min','temperatura_max_c':'max','precipitacion_mm':'sum','viento_vel_media_10m_ms':'mean','viento_u_media_10m_ms':'mean','viento_v_media_10m_ms':'mean','iith_h':'sum'}
    for window in [7,14]:
        features=frame[frame.dias_anteriores<=window].groupby('id_medicion').agg(aggregations)
        features=accepted.merge(features,on='id_medicion',validate='one_to_one').sort_values(['fecha_objetivo','id_medicion']).reset_index(drop=True)
        assert features[FEATURES+['iith_h']].notna().all().all()
        features.to_csv(PROCESSED/f'features_{window}dias.csv',index=False)
    split(features)
    save_json(REPORTS/'cobertura.json',{'mediciones_completas':len(accepted),'filas_14dias':len(frame),'filas_7dias':int((frame.dias_anteriores<=7).sum()),'mediciones_excluidas':len(targets)-len(accepted),'roya_historica':frame.estado_registro_roya.value_counts().to_dict(),'hmf':'pendiente'})

def split(features):
    dates=pd.to_datetime(features.fecha_objetivo)
    test_start=pd.Timestamp('2026-01-01')
    train_mask=dates<test_start-pd.Timedelta(days=14)
    test_mask=dates>=test_start
    train=features[train_mask]
    unique=sorted(train.fecha_objetivo.unique())
    assert len(unique)>21
    folds=[]
    for i in range(3):
        val_dates=unique[len(unique)-21+i*7:len(unique)-14+i*7]
        val_start=pd.Timestamp(val_dates[0])
        tr=features[dates<val_start-pd.Timedelta(days=14)]
        va=train[train.fecha_objetivo.isin(val_dates)]
        assert tr.objetivo_positivo.nunique()==va.objetivo_positivo.nunique()==2
        folds.append({'fold':i+1,'train_ids':tr.id_medicion.tolist(),'validation_ids':va.id_medicion.tolist(),'validation_dates':val_dates,'train_class_counts':tr.objetivo_positivo.value_counts().to_dict(),'validation_class_counts':va.objetivo_positivo.value_counts().to_dict()})
    assert features[test_mask].objetivo_positivo.nunique()==2
    save_json(PROCESSED/'particiones.json',{'seed':42,'threshold':.5,'purge_days':14,'test_start':'2026-01-01','train_ids':train.id_medicion.tolist(),'test_ids':features[test_mask].id_medicion.tolist(),'purged_ids':features[~train_mask&~test_mask].id_medicion.tolist(),'folds':folds})

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--download',action='store_true');args=parser.parse_args()
    build(args.download)
