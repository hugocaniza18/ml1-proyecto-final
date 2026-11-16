"""Ejecuta los notebooks de revisión con el mismo Python que inicia este script."""
import os, json, sys, shutil
from pathlib import Path
import nbformat
from nbclient import NotebookClient
ROOT=Path(__file__).resolve().parents[1]
kernel=ROOT/'work/jupyter/kernels/roya-tp'
kernel.mkdir(parents=True,exist_ok=True)
(kernel/'kernel.json').write_text(json.dumps({'argv':[sys.executable,'-m','ipykernel_launcher','-f','{connection_file}'],'display_name':'Roya TP','language':'python'}),encoding='utf-8')
os.environ['JUPYTER_PATH']=str(ROOT/'work/jupyter')
for p in sorted((ROOT/'notebooks').glob('*.ipynb')):
    nb=nbformat.read(p,as_version=4)
    NotebookClient(nb,timeout=180,kernel_name='roya-tp',resources={'metadata':{'path':str(ROOT)}}).execute()
    nbformat.write(nb,p)
    shutil.copy2(p,ROOT/'web/public/downloads'/p.name)
    print(f'{p.name}: ejecutado',flush=True)
