"""Ejecuta notebooks expresamente indicados; no publica archivos."""
import argparse
import json
import os
import sys
from pathlib import Path
import nbformat
from nbclient import NotebookClient
from procesar_datos import ROOT, require

def run(paths):
    kernel=ROOT/'.cache/jupyter/kernels/roya-tp'
    kernel.mkdir(parents=True,exist_ok=True)
    (kernel/'kernel.json').write_text(json.dumps({'argv':[sys.executable,'-m','ipykernel_launcher','-f','{connection_file}'], 'display_name':'Roya TP','language':'python'}),encoding='utf-8')
    os.environ['JUPYTER_PATH']=str(ROOT/'.cache/jupyter')
    for path in paths:
        require(path.resolve().is_relative_to((ROOT/'notebooks').resolve()), 'Notebook fuera del proyecto')
        nb=nbformat.read(path,as_version=4)
        NotebookClient(nb,timeout=180,kernel_name='roya-tp',resources={'metadata':{'path':str(ROOT)}}).execute()
        nbformat.write(nb,path)

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('notebooks',nargs='+',type=Path)
    run(parser.parse_args().notebooks)
