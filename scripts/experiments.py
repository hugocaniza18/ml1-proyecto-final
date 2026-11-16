"""Selección temporal en desarrollo, congelación y evaluación retrospectiva."""
import argparse, hashlib, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import joblib
from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, balanced_accuracy_score, roc_auc_score, average_precision_score, ConfusionMatrixDisplay, RocCurveDisplay, PrecisionRecallDisplay
from data_pipeline import ROOT, PROCESSED, REPORTS, FEATURES, save_json

plt.rcParams.update({'figure.dpi':120,'savefig.dpi':160,'axes.spines.top':False,'axes.spines.right':False,'font.size':10})
FIG=REPORTS/'figures'

def score(y,p):
    pred=(p>=.5).astype(int)
    return dict(f1=float(f1_score(y,pred,zero_division=0)),precision=float(precision_score(y,pred,zero_division=0)),recall=float(recall_score(y,pred,zero_division=0)),accuracy=float(accuracy_score(y,pred)),balanced_accuracy=float(balanced_accuracy_score(y,pred)),roc_auc=float(roc_auc_score(y,p)),average_precision=float(average_precision_score(y,p)))

def inputs(window):
    return pd.read_csv(PROCESSED/f'features_{window}dias.csv').set_index('id_medicion')

def eda():
    splits=json.loads((PROCESSED/'particiones.json').read_text())
    a=inputs(7).loc[splits['train_ids']];b=inputs(14).loc[splits['train_ids']]
    a[FEATURES+['iith_h']].describe().to_csv(REPORTS/'descriptivos_desarrollo_7dias.csv')
    b[FEATURES+['iith_h']].describe().to_csv(REPORTS/'descriptivos_desarrollo_14dias.csv')
    counts=a.resultado_objetivo.value_counts().reindex(['Negativo','Positivo'])
    fig,ax=plt.subplots(figsize=(7,4));ax.bar(counts.index,counts.values,color=['#19685b','#c77937']);ax.set(ylabel='Mediciones objetivo',title=f'Clases en desarrollo tras purga (n={len(a)})')
    for i,v in enumerate(counts):ax.text(i,v+1,str(v),ha='center')
    fig.tight_layout();fig.savefig(FIG/'clases.png');plt.close(fig)
    monthly=a.assign(mes=a.fecha_objetivo.str[:7]).groupby(['mes','resultado_objetivo']).size().unstack(fill_value=0)
    fig,ax=plt.subplots(figsize=(10,4));monthly.plot.bar(stacked=True,ax=ax,color=['#19685b','#c77937']);ax.set(ylabel='Mediciones',xlabel='Mes',title='Cobertura temporal de desarrollo');fig.tight_layout();fig.savefig(FIG/'tiempo.png');plt.close(fig)
    geo=a.groupby(['latitud','longitud']).size().reset_index(name='n')
    fig,ax=plt.subplots(figsize=(7,5));dots=ax.scatter(geo.longitud,geo.latitud,s=geo.n*6,c=geo.n,cmap='viridis',edgecolor='white');fig.colorbar(dots,ax=ax,label='Mediciones');ax.set(xlabel='Longitud (°)',ylabel='Latitud (°)',title='Puntos de monitoreo en desarrollo');fig.tight_layout();fig.savefig(FIG/'geografia.png');plt.close(fig)
    corr=a[FEATURES+['iith_h']].corr();corr.to_csv(REPORTS/'correlaciones_7dias.csv')
    labels=['HR media','HR mín.','HR máx.','T media','T mín.','T máx.','Precip.','Viento','Viento u','Viento v','IITH']
    fig,ax=plt.subplots(figsize=(8,7));im=ax.imshow(corr,vmin=-1,vmax=1,cmap='RdBu_r');ax.set_xticks(range(11),labels,rotation=60,ha='right');ax.set_yticks(range(11),labels);fig.colorbar(im,ax=ax,label='Correlación de Pearson');ax.set_title('Correlaciones en desarrollo, ventana de 7 días');fig.tight_layout();fig.savefig(FIG/'correlaciones.png');plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(12,4))
    for ax,col,label in zip(axes,['temperatura_media_c','precipitacion_mm','iith_h'],['Temperatura media (°C)','Precipitación acumulada (mm)','IITH acumulado (h)']):
        ax.boxplot([a[col],b[col]],tick_labels=['7 días','14 días']);ax.set_title(label)
    fig.tight_layout();fig.savefig(FIG/'ventanas.png');plt.close(fig)
    outliers={}
    for col in FEATURES+['iith_h']:
        q1,q3=a[col].quantile([.25,.75]);iqr=q3-q1
        outliers[col]=int(((a[col]<q1-1.5*iqr)|(a[col]>q3+1.5*iqr)).sum())
    pairs=corr.where(np.triu(np.ones(corr.shape),1).astype(bool)).stack()
    strongest=pairs.abs().idxmax()
    save_json(REPORTS/'eda.json',{'n_desarrollo':len(a),'clases':counts.to_dict(),'puntos':len(geo),'meses_con_registros':len(monthly),'extremos_iqr_no_eliminados':outliers,'correlacion_mayor':{'variables':list(strongest),'r':float(corr.loc[strongest])},'medianas_7dias':a[['temperatura_media_c','precipitacion_mm','iith_h']].median().to_dict(),'medianas_14dias':b[['temperatura_media_c','precipitacion_mm','iith_h']].median().to_dict()})

def train(evaluate=False):
    splits=json.loads((PROCESSED/'particiones.json').read_text());results=[];fitted={};fold_rows=[];learning=[]
    for window in [7,14]:
        data=inputs(window).loc[splits['train_ids']];y=data.objetivo_positivo
        folds=[(data.index.get_indexer(f['train_ids']),data.index.get_indexer(f['validation_ids'])) for f in splits['folds']]
        assert all((tr>=0).all() and (va>=0).all() for tr,va in folds)
        for iith in [False,True]:
            cols=FEATURES+(['iith_h'] if iith else []);x=data[cols]
            for model in ['LR','RF']:
                name=f'{model}_{window}d_'+('iith' if iith else 'base')
                if model=='LR':
                    estimator=Pipeline([('scaler',StandardScaler()),('model',LogisticRegression(random_state=42,max_iter=3000))]);grid={'model__C':[.1,1,10]}
                else:
                    estimator=Pipeline([('model',RandomForestClassifier(n_estimators=300,random_state=42,n_jobs=1))]);grid={'model__max_depth':[3,5,None],'model__min_samples_leaf':[3,5,10]}
                search=GridSearchCV(estimator,grid,cv=folds,scoring='f1',n_jobs=2,return_train_score=True,error_score='raise').fit(x,y)
                cv=pd.DataFrame(search.cv_results_);cv.to_csv(REPORTS/f'grid_{name}.csv',index=False)
                best=search.best_index_
                record={'config':name,'modelo':model,'dias':window,'iith':iith,'n_features':len(cols),'cv_f1':float(search.best_score_),'cv_std':float(search.cv_results_['std_test_score'][best]),'params':search.best_params_}
                results.append(record);fitted[name]=(search.best_estimator_,cols)
                joblib.dump({'pipeline':search.best_estimator_,'features':cols,'threshold':.5,'config':record},ROOT/'models'/f'{name}.joblib')
                for fold,(tr,va) in enumerate(folds,1):
                    est=clone(search.best_estimator_).fit(x.iloc[tr],y.iloc[tr]);p=est.predict_proba(x.iloc[va])[:,1]
                    fold_rows.append({'config':name,'fold':fold,'n_train':len(tr),'n_validation':len(va),**score(y.iloc[va],p)})
                    # Prefijos de fechas completas, sin mezclar fechas ni retirar la purga.
                    dates=sorted(data.iloc[tr].fecha_objetivo.unique())
                    for fraction in [.4,.7,1.]:
                        prefix=dates[:max(1,int(np.ceil(len(dates)*fraction)))];ix=tr[data.iloc[tr].fecha_objetivo.isin(prefix).to_numpy()]
                        if y.iloc[ix].nunique()!=2:continue
                        part=clone(search.best_estimator_).fit(x.iloc[ix],y.iloc[ix])
                        learning.append({'config':name,'fold':fold,'fraction':fraction,'n_train':len(ix),'train_f1':f1_score(y.iloc[ix],part.predict(x.iloc[ix]),zero_division=0),'validation_f1':f1_score(y.iloc[va],part.predict(x.iloc[va]),zero_division=0)})
                print(f"{name}: F1 CV={record['cv_f1']:.4f}",flush=True)
    winner=sorted(results,key=lambda r:(-round(r['cv_f1'],3),r['dias'],r['n_features'],0 if r['modelo']=='LR' else 1))[0]
    freeze={'selected':winner,'selection_rule':'F1 medio CV; empate a 3 decimales: 7d, menos variables, LR','threshold':.5,'split_sha256':hashlib.sha256((PROCESSED/'particiones.json').read_bytes()).hexdigest(),'data_sha256':{str(w):hashlib.sha256((PROCESSED/f'features_{w}dias.csv').read_bytes()).hexdigest() for w in [7,14]},'configs':results}
    freeze_path=REPORTS/'seleccion_congelada.json'
    if freeze_path.exists():
        assert json.loads(freeze_path.read_text())==freeze,'Selección distinta: no sobrescribir evaluación congelada.'
    else:save_json(freeze_path,freeze)
    pd.DataFrame(fold_rows).to_csv(REPORTS/'metricas_folds.csv',index=False)
    learn=pd.DataFrame(learning);learn.to_csv(REPORTS/'curvas_aprendizaje.csv',index=False)
    fig,axes=plt.subplots(2,4,figsize=(16,7),sharey=True)
    for ax,r in zip(axes.flat,results):
        subset=learn[learn.config==r['config']].groupby('fraction')[['n_train','train_f1','validation_f1']].mean()
        ax.plot(subset.n_train,subset.train_f1,'o-',label='Entrenamiento');ax.plot(subset.n_train,subset.validation_f1,'o-',label='Validación');ax.set(title=r['config'],ylim=(0,1.05),xlabel='Mediciones (media)');ax.legend(fontsize=7)
    fig.supylabel('F1 positivo');fig.tight_layout();fig.savefig(FIG/'aprendizaje.png');plt.close(fig)
    if not evaluate:return
    predictions=[]
    fig_cm,axes=plt.subplots(2,4,figsize=(16,7))
    fig_roc,rocax=plt.subplots(figsize=(8,6));fig_pr,prax=plt.subplots(figsize=(8,6))
    for r,ax in zip(results,axes.flat):
        test=inputs(r['dias']).loc[splits['test_ids']];est,cols=fitted[r['config']];p=est.predict_proba(test[cols])[:,1];y=test.objetivo_positivo
        r.update({'test_'+k:v for k,v in score(y,p).items()})
        for (mid,row),prob in zip(test.iterrows(),p):
            predictions.append({'id_medicion':mid,'fecha_objetivo':row.fecha_objetivo,'municipio':row.municipio,'resultado_objetivo':row.resultado_objetivo,'y_true':int(row.objetivo_positivo),'config':r['config'],'probabilidad_positivo':float(prob),'prediccion':int(prob>=.5)})
        ConfusionMatrixDisplay.from_predictions(y,p>=.5,normalize='true',display_labels=['Neg.','Pos.'],ax=ax,colorbar=False,cmap='Greens');ax.set_title(r['config'])
        RocCurveDisplay.from_predictions(y,p,name=r['config'],ax=rocax)
        PrecisionRecallDisplay.from_predictions(y,p,name=r['config'],ax=prax)
    fig_cm.tight_layout();fig_cm.savefig(FIG/'confusion.png');plt.close(fig_cm)
    rocax.set_title('ROC en prueba reservada');fig_roc.tight_layout();fig_roc.savefig(FIG/'roc.png');plt.close(fig_roc)
    prax.axhline(float(y.mean()),color='gray',ls='--',label='Prevalencia');prax.set_title('Precisión–recall en prueba reservada');prax.legend(fontsize=8);fig_pr.tight_layout();fig_pr.savefig(FIG/'precision_recall.png');plt.close(fig_pr)
    pd.DataFrame(results).to_csv(REPORTS/'comparacion_modelos.csv',index=False)
    pd.DataFrame(predictions).to_csv(REPORTS/'predicciones_prueba.csv',index=False)
    save_json(REPORTS/'resultados.json',{'selected_config':winner['config'],'n_train':len(splits['train_ids']),'n_test':len(splits['test_ids']),'n_purged':len(splits['purged_ids']),'results':results})

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--stage',choices=['eda','train','all'],default='all');parser.add_argument('--evaluate',action='store_true');args=parser.parse_args()
    if args.stage in ['eda','all']:eda()
    if args.stage in ['train','all']:train(args.evaluate)
