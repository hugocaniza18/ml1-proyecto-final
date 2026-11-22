"""Comprueba la migración sin ajustar modelos ni modificar resultados oficiales."""
import argparse
import json
import shutil
import tempfile
from pathlib import Path
import pandas as pd
from auditar_datos import verify as audit
from evaluar import evaluate
from procesar_datos import ROOT, configured, run, sha, require, write_json
from verificar import verify
from entrenar import validate_trace


def compare_csv(a, b, tolerance):
    left, right = pd.read_csv(a), pd.read_csv(b)
    pd.testing.assert_frame_equal(left, right, check_exact=False, atol=tolerance, rtol=0)


def integration():
    migration=json.loads((ROOT/'provenance/migracion.json').read_text(encoding='utf-8'))
    for entry in migration['archivos_conservados']:
        require(sha(ROOT/entry['destino'])==entry['sha256'], 'Artefacto aprobado modificado: '+entry['destino'])
    result=verify(configured('datos'), configured('resultados'))
    result['traza_sha256']=validate_trace(configured('datos'))
    with tempfile.TemporaryDirectory(prefix='roya-integracion-') as temp:
        temp=Path(temp)
        result['horarios']=audit(configured('era5'), configured('original').parents[1],
                                 configured('datos'), temp/'auditoria')
        run(configured('original').parents[1],temp/'procesamiento')
        for source in configured('datos').glob('*.csv'):
            compare_csv(source,temp/'procesamiento'/source.name,1e-10)
        for name in ['particiones.json','predictores.json']:
            require(json.loads((configured('datos')/name).read_text(encoding='utf-8')) ==
                    json.loads((temp/'procesamiento'/name).read_text(encoding='utf-8')), 'Particiones o predictores distintos')
        # Evaluate only the already-frozen pipelines in an isolated directory.
        output=temp/'results'
        (output/'predictions').mkdir(parents=True)
        shutil.copytree(configured('modelos'),temp/'models')
        for name in ['seleccion.json','secuencia.json']:
            shutil.copy2(configured('resultados')/name,output/name)
        timeline=json.loads((output/'secuencia.json').read_text(encoding='utf-8'))[:1]
        write_json(output/'secuencia.json',timeline)
        for relative in ['metricas.csv','predictions/predicciones.csv']:
            frame=pd.read_csv(configured('resultados')/relative)
            frame[frame.rol.eq('validacion')].to_csv(output/relative,index=False)
        evaluate(configured('datos'),output)
        for relative in ['metricas.csv','comparacion.csv','predictions/predicciones.csv']:
            compare_csv(configured('resultados')/relative,output/relative,1e-12)
        require(sha(output/'seleccion.json')==sha(configured('resultados')/'seleccion.json'), 'Selección alterada')
    result.update(procesamiento_reproducido=True,evaluacion_reproducida=True,
                  artefactos_inmutables=len(migration['archivos_conservados']),entrenamientos=0)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--save',type=Path)
    args=parser.parse_args()
    result=integration()
    if args.save:
        require(args.save.resolve()!= (configured('resultados')/'verificacion.json').resolve(), 'No sobrescribir evidencia original')
        write_json(args.save,result)
    print(json.dumps(result,ensure_ascii=False,indent=2))
