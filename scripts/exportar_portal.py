"""Prepara recursos agregados en un destino nuevo; no cambia el portal."""
import argparse
import shutil
from pathlib import Path
from procesar_datos import configured, require

# Lista explícita: nunca recorre specs/, fuentes, modelos o predicciones individuales.
TABLES = ['comparacion.csv', 'metricas.csv', 'aprendizaje.csv', 'diferencias_descriptivas.csv']


def export(destination):
    require(not destination.exists(), 'Destino existente; no sobrescribir el portal')
    destination.mkdir(parents=True)
    for name in TABLES:
        shutil.copy2(configured('resultados')/name,destination/name)
    shutil.copytree(configured('informe')/'figuras',destination/'figuras')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--destination',type=Path,required=True)
    export(parser.parse_args().destination)
