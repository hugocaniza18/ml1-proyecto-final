"""Exporta documentación y portal desde los resultados, sin volver a seleccionar modelos."""
import json, shutil
import pandas as pd
import nbformat as nbf
from data_pipeline import ROOT, PROCESSED, REPORTS, FEATURES, save_json

def table_md(df):
    cols=list(df.columns)
    rows=['| '+' | '.join(cols)+' |','| '+' | '.join(['---']*len(cols))+' |']
    rows += ['| '+' | '.join(str(v) for v in row)+' |' for row in df.itertuples(index=False,name=None)]
    return '\n'.join(rows)

def build():
    if (ROOT/'web/public/downloads/informe-con-figuras.zip').exists():
        raise SystemExit('Regenera los informes en el paquete local. Esta edición pública usa una exportación sin identificadores de origen.')
    result=json.loads((REPORTS/'resultados.json').read_text());eda=json.loads((REPORTS/'eda.json').read_text());audit=json.loads((REPORTS/'auditoria.json').read_text());coverage=json.loads((REPORTS/'cobertura.json').read_text());split=json.loads((PROCESSED/'particiones.json').read_text());team=json.loads((ROOT/'team.json').read_text())
    selected=next(r for r in result['results'] if r['config']==result['selected_config'])
    comparison=pd.DataFrame(result['results'])
    pairs=[]
    for model in ['LR','RF']:
        for iith in [False,True]:
            p=comparison[(comparison.modelo==model)&(comparison.iith==iith)].set_index('dias')
            pairs.append({'modelo':model,'IITH':iith,'delta_F1_CV_14_menos_7':float(p.loc[14,'cv_f1']-p.loc[7,'cv_f1']),'delta_F1_prueba_14_menos_7':float(p.loc[14,'test_f1']-p.loc[7,'test_f1'])})
    pd.DataFrame(pairs).to_csv(REPORTS/'comparacion_ventanas.csv',index=False)
    conclusions=[f"La selección por validación fue {selected['config']}, con F1 medio {selected['cv_f1']:.3f} y desviación entre bloques {selected['cv_std']:.3f}.",f"En prueba, su F1 fue {selected['test_f1']:.3f}, precisión {selected['test_precision']:.3f} y recall {selected['test_recall']:.3f}. Detectó 13 de los 37 positivos y omitió 24. Clasificó correctamente 24 de los 29 negativos.","Ampliar a 14 días no mostró una mejora consistente: el F1 de validación bajó en tres de cuatro comparaciones equivalentes. En prueba mejoró solo LR sin IITH. Son diferencias descriptivas, sin evidencia de significancia estadística.","RF con 7 días e IITH logró el mayor F1 de prueba (0,481), pero no reemplaza al modelo seleccionado previamente. RF con 14 días e IITH logró el mayor ROC-AUC (0,753): el ordenamiento de probabilidades y el F1 con umbral 0,5 miden propiedades diferentes.","Agregar IITH no mejoró F1 de validación en ninguna comparación equivalente. Su utilidad futura no queda descartada por este conjunto pequeño.","La caída entre validación y prueba señala generalización temporal limitada. Las curvas de aprendizaje muestran diferencias entre ajuste y validación, compatibles con sobreajuste y cambios de distribución. No permiten separar sus causas."]
    captions={
        'clases.png':f"Desarrollo después de la purga: {eda['clases']['Negativo']} negativos y {eda['clases']['Positivo']} positivos. Las clases son cercanas en tamaño, lo que no garantiza equilibrio por fecha o punto.",
        'tiempo.png':'Hay registros en diez meses de desarrollo y un intervalo largo sin observaciones. Ausencia de registros no equivale a ausencia de roya.',
        'geografia.png':'Los 22 puntos tienen cantidades distintas de observaciones. El tamaño representa frecuencia de monitoreo, no riesgo ni área cultivada.',
        'correlaciones.png':f"La mayor correlación absoluta entre predictores de 7 días es entre temperatura media y máxima (r={eda['correlacion_mayor']['r']:.3f}). Es redundancia meteorológica, no evidencia causal de roya.",
        'ventanas.png':f"Las medianas de precipitación son {eda['medianas_7dias']['precipitacion_mm']:.1f} y {eda['medianas_14dias']['precipitacion_mm']:.1f} mm. IITH: {eda['medianas_7dias']['iith_h']:.0f} y {eda['medianas_14dias']['iith_h']:.0f} h. Los acumulados crecen con la duración de ventana, lo que no demuestra mejor predicción.",
        'aprendizaje.png':'Promedios de tres folds con prefijos de fechas completas. El eje horizontal es el tamaño medio de entrenamiento. Son diagnósticos de desarrollo, no una estimación independiente del rendimiento seleccionado.',
        'confusion.png':'Filas normalizadas por clase real. El modelo seleccionado reconoce el 35,1 % de positivos y el 82,8 % de negativos; la sensibilidad a positivos es limitada.',
        'roc.png':'La ROC compara sensibilidad y falsas alarmas al variar el umbral. No se utilizó para cambiar el umbral fijado en 0,5.',
        'precision_recall.png':'La curva precisión–recall centra la atención en positivos. La línea horizontal es la prevalencia en prueba (37/66), no un modelo adicional entrenado.'}
    preview=comparison[['config','cv_f1','cv_std','test_f1','test_precision','test_recall','test_accuracy','test_balanced_accuracy','test_roc_auc','test_average_precision']].copy()
    for c in preview.columns[1:]:preview[c]=preview[c].map(lambda v:f'{v:.3f}')
    fold_summary=[{'fold':f['fold'],'entrenamiento':len(f['train_ids']),'validacion':len(f['validation_ids']),'inicio':f['validation_dates'][0],'fin':f['validation_dates'][-1]} for f in split['folds']]
    team_text='\n'.join(f"- {m['name']} — {m['email']}"+(f" — {m['github']}" if m['github'] else ' — perfil GitHub pendiente') for m in team['members'])
    report=f"# Predicción de roya de soja con datos climáticos\n\nMachine Learning 1 (23433), FIUNA, Ingeniería Electrónica, 2026.\n\n{team_text}\n\n## Problema, datos y metodología\n\n"+(ROOT/'docs/METODOLOGIA.md').read_text(encoding='utf-8').split('\n',1)[1]+f"\n\n## Auditoría y cobertura\n\n428 objetivos, 211 positivos y 217 negativos, 22 coordenadas y 47 fechas. Correspondencia completa con el original. Las ventanas contienen 5.992 y 2.996 filas, pero cada tabla de aprendizaje tiene 428 muestras. No hubo exclusiones por clima faltante. En 14 días hay 684 antecedentes evaluados, 79 no evaluados y 5.229 sin registro exacto.\n\n## Particiones utilizadas\n\n{table_md(pd.DataFrame(fold_summary))}\n\nPrueba: 66 mediciones (37 positivas y 29 negativas), enero–marzo de 2026. Purga final: 3.\n\n## Análisis exploratorio\n\n"
    for file,caption in captions.items():
        if file in ['aprendizaje.png','confusion.png','roc.png','precision_recall.png']:continue
        report+=f'![{caption}](figures/{file})\n\n{caption}\n\n'
    report+='## Resultados\n\n'+table_md(preview)+'\n\nF1, precisión y recall se refieren a la clase positiva. La dispersión CV corresponde a tres bloques temporales.\n\n### Efecto de la ventana\n\n'+table_md(pd.DataFrame(pairs).round(4))+'\n\n'
    for file in ['aprendizaje.png','confusion.png','roc.png','precision_recall.png']:report+=f'![{captions[file]}](figures/{file})\n\n{captions[file]}\n\n'
    report+='## Conclusiones\n\n'+'\n\n'.join(conclusions)+'\n\n## Trabajo futuro y entrega\n\nValidar nuevas campañas y regiones; confirmar protocolo de etiqueta y disponibilidad temporal; incorporar HMF, estaciones y contexto agronómico. Explorar calibración y umbrales en un nuevo desarrollo, conservando una prueba futura intacta.\n\nDatos meteorológicos de Open-Meteo (ERA5), CC BY 4.0.\n'
    (REPORTS/'INFORME.md').write_text(report,encoding='utf-8')
    dictionary=[]
    frame=pd.read_csv(PROCESSED/'monitoreo_ml_14dias.csv')
    meanings={'id_medicion':'Identificador estable del objetivo; agrupar por él','latitud':'Latitud WGS84 del monitoreo','longitud':'Longitud WGS84 del monitoreo','fecha_objetivo':'Fecha del resultado publicado','resultado_objetivo':'Etiqueta CIRAM, Positivo o Negativo','dias_anteriores':'Distancia entera al objetivo: 1 a 14','fecha_dato':'Fecha local anterior al objetivo','roya_anterior':'Positivo/Negativo auxiliar; vacío si no evaluado o sin registro','estado_registro_roya':'evaluado, no_evaluado o sin_registro','iith_h':'Horas HR>=85 y 15<=T<=28, límites inclusivos','horas_hr_ge90_h':'Horas HR>=90; auxiliar, no HMF','viento_dir_10m_grados':'Dirección vectorial de procedencia; vacío si vector medio nulo'}
    for c in frame:
        unit='%' if c.startswith('hr_') else '°C' if c.startswith('temperatura_') else 'mm' if c.endswith('_mm') else 'm/s' if c.endswith('_ms') else 'h' if c.endswith('_h') else 'grados' if c in ['latitud','longitud','viento_dir_10m_grados'] else '—'
        dictionary.append({'columna':c,'tipo':str(frame[c].dtype),'unidad':unit,'descripcion':meanings.get(c,c.replace('_',' ')),'uso_modelo':'resumen de ventana' if c in FEATURES else 'variante IITH' if c=='iith_h' else 'etiqueta' if c=='resultado_objetivo' else 'auxiliar, excluido'})
    pd.DataFrame(dictionary).to_csv(ROOT/'docs/diccionario_datos.csv',index=False)
    (ROOT/'docs/DICCIONARIO.md').write_text('# Diccionario del archivo largo\n\n'+table_md(pd.DataFrame(dictionary))+'\n\nEn features_7dias y features_14dias se agregan medias como medias de días completos; mínimos/máximos como extremos; precipitación e IITH como sumas. objetivo_positivo: 1=Positivo, 0=Negativo. municipio y punto_id solo identifican. El archivo procedencia.csv conserva la celda ERA5, zona horaria y pendientes.\n',encoding='utf-8')
    public=ROOT/'web/public';(public/'figures').mkdir(exist_ok=True);(public/'downloads').mkdir(exist_ok=True)
    for f in (REPORTS/'figures').glob('*.png'):shutil.copy2(f,public/'figures'/f.name)
    for src in [ROOT/'docs/METODOLOGIA.md',ROOT/'docs/DICCIONARIO.md',REPORTS/'comparacion_modelos.csv',REPORTS/'comparacion_ventanas.csv',REPORTS/'metricas_folds.csv']:shutil.copy2(src,public/'downloads'/src.name)
    # Informe con figuras accesibles en el mismo paquete descargable se construye aparte.
    shutil.copy2(REPORTS/'INFORME.md',public/'downloads/INFORME.md')
    predictions=pd.read_csv(REPORTS/'predicciones_prueba.csv').to_dict('records')
    save_json(ROOT/'web/app/results.json',{**result,'audit':audit,'coverage':coverage,'eda':eda,'team':team,'folds':fold_summary,'captions':captions,'conclusions':conclusions,'window_comparison':pairs,'predictions':predictions})
    notebooks(captions,conclusions)
    for f in (ROOT/'notebooks').glob('*.ipynb'):shutil.copy2(f,public/'downloads'/f.name)

def notebooks(captions, conclusions):
    setup="from pathlib import Path\nimport json, sys\nimport pandas as pd\nfrom IPython.display import display, Image, Markdown\nROOT = Path.cwd() if (Path.cwd()/'data').exists() else Path.cwd().parent\nsys.path.insert(0, str(ROOT/'scripts'))\nfrom data_pipeline import audit\n"
    def notebook(name,cells):
        nb=nbf.v4.new_notebook(cells=[nbf.v4.new_markdown_cell(f'# {name.replace("_"," ")}'),nbf.v4.new_code_cell(setup)]+cells,metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'}})
        nbf.write(nb,ROOT/'notebooks'/f'{name}.ipynb')
    notebook('01_Auditoria',[
        nbf.v4.new_markdown_cell('La unidad de aprendizaje es la medición objetivo. Se verifica identidad contra los 650 registros originales y se excluyen los no evaluados.'),
        nbf.v4.new_code_cell("targets, original = audit()\ndisplay(pd.Series(json.loads((ROOT/'reports/auditoria.json').read_text())))\ndisplay(targets.groupby('resultado_objetivo').size())"),
        nbf.v4.new_code_cell("long = pd.read_csv(ROOT/'data/processed/monitoreo_ml_14dias.csv')\nassert not long.duplicated(['id_medicion','dias_anteriores']).any()\nassert long.groupby('id_medicion').size().eq(14).all()\ndisplay(long.estado_registro_roya.value_counts())\ndisplay(pd.Series(json.loads((ROOT/'reports/cobertura.json').read_text())))"),
        nbf.v4.new_markdown_cell('El cruce usa exactamente coordenadas y fecha. Sin registro no significa negativo. Los snapshots no son muestreos diarios. Véase docs/METODOLOGIA.md para el alcance de la etiqueta y fuentes.')])
    cells=[nbf.v4.new_markdown_cell('EDA limitado a los 359 objetivos de desarrollo tras la purga. Las decisiones no usan la prueba reservada.'),nbf.v4.new_code_cell("splits=json.loads((ROOT/'data/processed/particiones.json').read_text())\nx=pd.read_csv(ROOT/'data/processed/features_7dias.csv').set_index('id_medicion').loc[splits['train_ids']]\nassert len(x)==359\ndisplay(x.describe())")]
    for f in ['clases.png','tiempo.png','geografia.png','correlaciones.png','ventanas.png']:
        cells += [nbf.v4.new_code_cell(f"display(Image(filename=str(ROOT/'reports/figures/{f}')))"),nbf.v4.new_markdown_cell(captions[f])]
    notebook('02_EDA',cells)
    notebook('03_Modelado_y_evaluacion',[
        nbf.v4.new_markdown_cell('Experimentos reproducibles: ejecutar `python scripts/experiments.py --evaluate` desde la raíz. Este notebook revisa los artefactos ya calculados, verifica sus métricas y no vuelve a ajustar después de observar prueba. GridSearchCV, selección y congelación están implementados en scripts/experiments.py.'),
        nbf.v4.new_code_cell("frozen=json.loads((ROOT/'reports/seleccion_congelada.json').read_text())\ndisplay(pd.Series(frozen['selected']))\ndisplay(pd.read_csv(ROOT/'reports/comparacion_modelos.csv'))"),
        nbf.v4.new_code_cell("from verify import verify\nverify()\ndisplay(pd.read_csv(ROOT/'reports/comparacion_ventanas.csv'))\ndisplay(pd.read_csv(ROOT/'reports/metricas_folds.csv'))"),
        *[cell for f in ['aprendizaje.png','confusion.png','roc.png','precision_recall.png'] for cell in [nbf.v4.new_code_cell(f"display(Image(filename=str(ROOT/'reports/figures/{f}')))"),nbf.v4.new_markdown_cell(captions[f])]],
        nbf.v4.new_markdown_cell('## Conclusiones\n\n'+'\n\n'.join(conclusions))])

if __name__=='__main__':build()
