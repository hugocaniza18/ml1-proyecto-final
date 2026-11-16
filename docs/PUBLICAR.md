# Publicar ml1-proyecto-final

Esta carpeta contiene la edición pública. Conserva el paquete local completo para reproducir el análisis y consultar los CSV de origen.

## 1. GitHub

1. Inicia sesión en GitHub con la cuenta `hugocaniza18`.
2. Abre el [repositorio existente](https://github.com/hugocaniza18/ml1-proyecto-final) y comprueba su visibilidad y si ya contiene commits.
3. Abre PowerShell en esta carpeta, la que contiene `team.json` y `web`. Ejecuta:

```powershell
git init -b main
git add .
git diff --cached --name-only
git commit -m "Prepara la entrega final de Machine Learning 1"
git remote add origin https://github.com/hugocaniza18/ml1-proyecto-final.git
git push -u origin main
```

Si Git pide identificar el autor, configura `git config user.name "Hugo Rene Caniza Portillo"` y `git config user.email "TU_CORREO_DE_GITHUB"` en esta carpeta. Puedes usar la dirección noreply que aparece en GitHub → Settings → Emails. Repite desde el commit. Si Git abre una ventana de autenticación, complétala en el navegador. Cuando ya exista un remoto, revisa `git remote -v` antes de cambiarlo. Si el remoto ya tiene commits, conserva su historial y no fuerces el push.

## 2. Vercel

1. Entra en https://vercel.com/new y autentícate. En **Import Git Repository**, elige **Import** junto a `hugocaniza18/ml1-proyecto-final`. Si el repositorio no aparece, concede acceso a ese repositorio en la integración de GitHub de Vercel.
2. En la configuración usa **Next.js**, **Root Directory** `web` y Node.js **24.x**. Deja que Vercel detecte **Output Directory**. El proyecto usa `npm run build` y no necesita variables de entorno. Confirma **Production Branch** `main` en los ajustes del proyecto.
3. Si quieres revisar la compilación local primero, ejecuta `npm ci` y `npm run build` desde `web`. Para tener Preview antes de producción, sube primero una rama distinta de `main` al proyecto conectado o usa un despliegue de Preview de Vercel CLI. Si importas directamente `main` y pulsas **Deploy**, ese primer despliegue será de producción.
4. Abre la URL de producción en incógnito y confirma que puedes navegar, cambiar las opciones del comparador y descargar el informe. Revisa la versión móvil. Entrega la URL de producción y la del repositorio.

## 3. Enlace de entrega

Los dos perfiles ya están en `team.json`. Registra la URL definitiva en su campo `deployment` y en el README. Haz commit y push. En Vercel, comprueba el commit servido en producción y abre el portal sin iniciar sesión.

## 4. Consulta a CIRAM

Usa la [página oficial de solicitud de datos](https://ciram.epagri.sc.gov.br/index.php/solucoes/laudos-dados/) para consultar por el conjunto fitosanitario de Agroconnect en particular. Puedes enviar este texto:

> Olá. Somos estudantes da Faculdade de Engenharia da Universidad Nacional de Asunción (FIUNA). Em um trabalho acadêmico analisamos registros públicos do monitoramento da ferrugem asiática da soja no Agroconnect, de setembro de 2024 a março de 2026, incluindo data, localização e resultado Positivo/Negativo. O endpoint de origem é `https://ciram.epagri.sc.gov.br/agroconnect/busca-doenca-munic.jsp`. Existe uma licença que permita redistribuir o CSV original e a versão derivada em um repositório público do GitHub? Quais são as condições de atribuição e de uso? Agradecemos a orientação.

Mientras tanto, publica esta edición con métricas, gráficos agregados y ejemplos sin fecha, municipio ni identificador de origen. Conserva el ZIP local para la reproducción. No hay una licencia de redistribución confirmada para los CSV del monitoreo.
