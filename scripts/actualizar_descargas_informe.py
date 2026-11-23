"""Sincroniza las descargas con el informe editado, sin regenerar su contenido."""
from pathlib import Path
import re
import base64
import shutil
import zipfile
import mistune

ROOT = Path(__file__).resolve().parents[1]


def run():
    source = ROOT/'reports/INFORME.md'
    public = ROOT/'web/public/academico'
    public.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, public/'INFORME.md')
    for notebook in (ROOT/'notebooks').glob('*.ipynb'):
        shutil.copy2(notebook, public/notebook.name)
    (public/'figuras').mkdir(exist_ok=True)
    render = mistune.create_markdown(escape=False, plugins=['table'])
    html = render(source.read_text(encoding='utf-8'))
    for filename in set(re.findall(r'src="figuras/([\w.-]+\.png)"', html)):
        image = ROOT/'reports/figuras'/filename
        shutil.copy2(image, public/'figuras'/filename)
        html = html.replace('src="figuras/'+image.name+'"',
            'src="data:image/png;base64,'+base64.b64encode(image.read_bytes()).decode()+'"')
    page = '<!doctype html><html lang="es"><meta charset="utf-8"><title>Roya de soja y clima</title><style>body{font:16px/1.6 Arial;max-width:1100px;margin:40px auto;padding:20px}img{max-width:100%}table{font-size:12px;border-collapse:collapse}td,th{padding:8px;border:1px solid #ccc}</style><body>'+html+'</body></html>'
    (public/'INFORME.html').write_text(page, encoding='utf-8')
    with zipfile.ZipFile(public/'informe-academico.zip', 'w', compression=zipfile.ZIP_DEFLATED) as archive:
        archive.write(public/'INFORME.html', arcname='INFORME.html')


if __name__ == '__main__':
    run()
