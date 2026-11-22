"""Genera informe y portal desde notebooks ejecutados; no ajusta modelos."""
from pathlib import Path
import nbformat,base64,json,shutil
import mistune
import re
from io import StringIO
import pandas as pd
from bs4 import BeautifulSoup
R=Path(__file__).resolve().parents[1]
render=mistune.create_markdown(escape=False,plugins=['table'])
figures=R/'reports/figuras'; figures.mkdir(exist_ok=True)
def excerpt(name,indices):
 n=nbformat.read(R/'notebooks'/name,4);parts=[]
 for i in indices:
  c=n.cells[i]
  if c.cell_type=='markdown':parts.append(c.source)
  else:
   for j,o in enumerate(c.outputs):
    if name.startswith('03') and i == 9 and j > 1: continue
    d=o.get('data',{})
    if 'image/png' in d:
     filename=f'{name[:2]}_{i}_{j}.png';(figures/filename).write_bytes(base64.b64decode(d['image/png']))
     caption = re.sub(r'^#+\s*\d*\.?\s*', '', next((n.cells[k].source.split('\n')[0] for k in range(i-1,-1,-1) if n.cells[k].cell_type=='markdown' and n.cells[k].source.startswith('#')), 'Resultados'))
     parts.append(f'![{caption} — panel {j+1}](figuras/{filename})\n\n*{caption}. Resultados oficiales del notebook académico.*')
    elif 'text/html' in d and '<table' in d['text/html']:
     if name.startswith('03') and i == 24: continue
     soup = BeautifulSoup(d['text/html'], 'html.parser')
     headers = [x.get_text(strip=True) for x in soup.thead.find_all('tr')[-1].find_all('th')]
     values = [[x.get_text(strip=True) for x in row.find_all('td')] for row in soup.tbody.find_all('tr')]
     if values and len(headers)==len(values[0])+1: headers=headers[1:]
     if not values or len(headers)!=len(values[0]) or len(set(headers))!=len(headers):
      parts.append(d['text/html']); continue
     table = pd.DataFrame(values, columns=headers)
     for col in table:
      numeric = pd.to_numeric(table[col], errors='coerce')
      if numeric.notna().all(): table[col] = numeric
     if name.startswith('03') and i == 9 and 'rol' in table.columns:
      table = table[table.rol.eq('validacion')]
     if name.startswith('03') and i == 27 and 'comparacion' in table.columns:
      table = table[table.comparacion.eq('14d menos 7d')]
     if 'historico_f1' in table.columns:
      keep = ['config','seleccionado','cv_f1','historico_f1','historico_precision','historico_recall','historico_roc_auc']
      table = table[[c for c in keep if c in table.columns]]
     elif 'rol' in table.columns:
      keep = ['horizonte','bloque','n','negativos','positivos','ubicaciones','etiqueta_inicio','etiqueta_fin']
      table = table[[c for c in keep if c in table.columns]]
     table = table.loc[:, ~table.columns.astype(str).str.startswith('Unnamed')]
     parts.append(table.to_html(index=False, float_format=lambda v: f'{v:.3f}'))
 return '\n\n'.join(parts)
team=json.loads((R/'team.json').read_text(encoding='utf-8'))
identity='\n\n'.join(f"{m['name']} · {m['email']} · [GitHub]({m['github']})" for m in team['members'])
sections=[('portada','1. Identificación académica',f'''# Roya de soja y clima

Machine Learning 1 (23433) · Ingeniería Electrónica · FIUNA, Universidad Nacional de Asunción · 2026.

{identity}

[Repositorio del proyecto]({team['repository']})

Evaluamos dos horizontes y dos ventanas meteorológicas mediante clasificación binaria. El estudio es retrospectivo y no constituye una alerta agronómica validada.'''),
('problema','2. Problema y objetivos','''La identificación del estado sanitario de la soja requiere seguimiento. Estudiamos si resúmenes de humedad, temperatura, lluvia y viento permiten discriminar el resultado registrado de roya asiática. No inferimos causalidad ni sustituimos una evaluación de campo.

Nuestro objetivo general es evaluar esa capacidad predictiva bajo separación temporal. Los objetivos específicos son auditar las etiquetas, comparar ventanas de 7 y 14 días, contrastar regresión logística y Random Forest y evaluar referencias sencillas de mayoría y persistencia.

**Entrada:** diez variables meteorológicas de t−7 a t−1 o de t−14 a t−1. **Salida h0:** estado registrado en t. **Salida h7:** estado registrado exactamente en t+7. H7 incluye positivos previos y no identifica exclusivamente primeras detecciones. Comparamos aprendizaje con referencias para comprobar si aporta valor frente a reglas sencillas.'''),
('datos','3. Datos y exploración',excerpt('01_auditoria.ipynb',[2,7,10,11,12,13,14,15,16,19,20,21,22])+ '\n\n'+excerpt('02_exploracion.ipynb',[4,5,6,7,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25])),
('metodologia','4. Metodología',excerpt('03_evaluacion.ipynb',[5,6,7,8,9,10,11,12,13])),
('resultados','5. Resultados y discusión',excerpt('03_evaluacion.ipynb',list(range(14,28)))),
('conclusiones','6. Conclusiones y referencias',excerpt('03_evaluacion.ipynb',[28]))]
# Add verified primary references without deriving new experimental claims.
sections[-1]=(sections[-1][0],sections[-1][1],sections[-1][2]+'''

### Bibliografía

Pedregosa, F. et al. (2011). Scikit-learn: Machine Learning in Python. *Journal of Machine Learning Research*, 12, 2825–2830. https://jmlr.org/papers/v12/pedregosa11a.html

Hersbach, H. et al. (2020). The ERA5 global reanalysis. *Quarterly Journal of the Royal Meteorological Society*, 146, 1999–2049. https://doi.org/10.1002/qj.3803

La ejecución de auditoría y EDA requiere acceso autorizado a los registros originales y tablas procesadas locales. La evaluación académica utiliza resultados agregados públicos. No redistribuimos registros individuales, modelos ni predicciones por muestra. Los notebooks contienen sus resultados visibles y documentan sus requisitos de ejecución.

El portal se despliega en Vercel, plataforma cuya aceptación fue confirmada por la cátedra.''')
id,title,body=sections[4]
sections[4]=(id,title,body+'\n\n### Matrices normalizadas\n\nCada fila suma uno y expresa qué proporción de cada clase real se clasifica correctamente. La normalización facilita comparar clases de diferente tamaño; los conteos anteriores conservan el tamaño efectivo del histórico.\n\n![Matrices normalizadas h0](figuras/confusion_h0.png)\n\n![Matrices normalizadas h7](figuras/confusion_h7.png)')
# Unificar la jerarquía del informe y evitar repetir notas de reproducción.
sections[1]=(sections[1][0],sections[1][1],sections[1][2]+"\n\nLa roya asiática es causada por *Phakopsora pachyrhizi*. La temperatura y la humedad sobre la superficie foliar intervienen en la infección, mientras la meteorología también afecta dispersión y desarrollo epidémico (Del Ponte y Esker, 2008; Melching et al., 1989). Esto motiva estudiar variables climáticas, sin equiparar humedad relativa con mojado foliar ni extrapolar resultados controlados directamente a una alerta de campo.")
sections[-1]=(sections[-1][0],sections[-1][1],sections[-1][2]+"\n\nDel Ponte, E. M. y Esker, P. D. (2008). Meteorological factors and Asian soybean rust epidemics: a systems approach and implications for risk assessment. *Scientia Agricola*, 65, 88–97. [Artículo](https://www.scielo.br/j/sa/a/DnFFn5X4cLyTZyKLHQ4YbLt/?lang=en).\n\nMelching, J. S. et al. (1989). Effects of duration, frequency, and temperature of leaf wetness periods on soybean rust. *Plant Disease*, 73, 117–122. [Artículo original](https://www.apsnet.org/publications/PlantDisease/BackIssues/Documents/1989Articles/PlantDisease73n02_117.PDF).")
normalized=[]
for number,(id,title,body) in enumerate(sections,1):
 if number==6:
  body=re.sub(r'## Fuentes y reproducción.*?(?=### Bibliografía)', '', body, flags=re.S)
 counter=[0]
 def heading(match):
  counter[0]+=1
  text=re.sub(r'^\d+\.\s*','',match.group(1))
  return f'### {number}.{counter[0]} {text}'
 body=re.sub(r'^#{2,3} (.+)$',heading,body,flags=re.M)
 body=body.replace('# Roya de soja y clima','**Roya de soja y clima**')
 normalized.append((id,title,body))
sections=normalized
report='\n\n'.join(f'## {title}\n\n{body}' for _,title,body in sections)
(R/'reports/INFORME.md').write_text(report,encoding='utf-8')
public=R/'web/public/academico';public.mkdir(exist_ok=True)
shutil.copytree(figures,public/'figuras',dirs_exist_ok=True)
shutil.copy2(R/'reports/INFORME.md',public/'INFORME.md')
html=render(report)
for p in figures.glob('*.png'):
 html=html.replace('src="figuras/'+p.name+'"','src="data:image/png;base64,'+base64.b64encode(p.read_bytes()).decode()+'"')
(public/'INFORME.html').write_text('<!doctype html><html lang="es"><meta charset="utf-8"><title>Roya de soja y clima</title><style>body{font:16px/1.6 Arial;max-width:1100px;margin:40px auto;padding:20px}img{max-width:100%}table{font-size:12px;border-collapse:collapse}td,th{padding:8px;border:1px solid #ccc}</style><body>'+html+'</body></html>',encoding='utf-8')
for p in (R/'notebooks').glob('*.ipynb'):shutil.copy2(p,public/p.name)
(R/'web/app/academico.json').write_text(json.dumps([dict(id=id,title=title,html=render(body).replace('src="figuras/','src="/academico/figuras/')) for id,title,body in sections],ensure_ascii=False),encoding='utf-8')
(R/'web/app/page.tsx').write_text('''import sections from './academico.json';
export default function Home(){return <><header><div className="wrap"><strong>Roya de soja y clima</strong><nav>{sections.map(s=><a key={s.id} href={'#'+s.id}>{s.title}</a>)}</nav></div></header><main className="wrap"><p className="notice">Estudio retrospectivo · h7 experimental · Evaluación histórica previamente observada</p>{sections.map(s=><section id={s.id} key={s.id}><h2>{s.title}</h2><article dangerouslySetInnerHTML={{__html:s.html}} /></section>)}<section><h2>Material académico</h2>{['INFORME.md','01_auditoria.ipynb','02_exploracion.ipynb','03_evaluacion.ipynb'].map(n=><p key={n}><a href={'/academico/'+n} download>{n}</a></p>)}</section></main><footer className="wrap">FIUNA · Machine Learning 1 · 2026</footer></>}
''',encoding='utf-8')
page=R/'web/app/page.tsx'
page.write_text(page.read_text(encoding='utf-8').replace("['INFORME.md',","['INFORME.html','INFORME.md',"),encoding='utf-8')
(R/'web/app/globals.css').write_text('''@import "tailwindcss";
:root{font-family:Arial,sans-serif;color:#18382e;background:#f7faf7}body{margin:0;line-height:1.65}.wrap{max-width:1120px;width:92%;margin:auto}header{background:#173d31;color:white;padding:20px 0}nav{display:flex;flex-wrap:wrap;gap:16px;font-size:13px}nav a{color:white}section{padding:36px 0;border-bottom:1px solid #ccd8ce;scroll-margin-top:20px}h1{font-size:40px}h2{font-size:28px}h3{font-size:21px}a{color:#24684e;text-decoration:underline}article{overflow-x:auto}article img{max-width:100%;height:auto;background:white}table{border-collapse:collapse;font-size:13px;width:100%;margin:24px 0}td,th{padding:10px;text-align:left;border-bottom:1px solid #d2ddd4}th{background:#e2eee4}p,li{max-width:90ch}.notice{background:#e3eee5;padding:18px}footer{padding:30px 0}pre{overflow:auto}@media(max-width:640px){h1{font-size:30px}h2{font-size:23px}td,th{padding:6px}section{padding:24px 0}}
''',encoding='utf-8')
print('Informe y portal académico preparados')
