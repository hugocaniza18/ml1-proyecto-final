# Fuentes, atribución y redistribución

Consulta documental: 2026-11-15.

- Etiquetas: Epagri/CIRAM Agroconnect, endpoint `https://ciram.epagri.sc.gov.br/agroconnect/busca-doenca-munic.jsp`. CSV suministrado por el usuario. Se conserva copia y hash. No se encontró en las páginas consultadas una licencia explícita del conjunto de etiquetas: **redistribución del dataset CIRAM pendiente de aclaración**. El acceso público por sí solo no establece una licencia. Los CSV originales y derivados quedan locales y excluidos por defecto de Git; no se les aplica una licencia inventada.
- Meteorología: **Weather data by Open-Meteo.com**, ERA5. [API](https://open-meteo.com/), [licencia CC BY 4.0](https://creativecommons.org/licenses/by/4.0/), [condiciones](https://open-meteo.com/en/terms), [licencias y proveedores](https://open-meteo.com/en/license). Transformaciones: agregación diaria y de ventanas, cálculo vectorial de viento e IITH. No implica aval del proveedor.
- Diseño académico: [Machine-Learning, carlobeni](https://github.com/carlobeni/Machine-Learning), ramas project-template, project-test y project-web, usadas como referencia estructural. Código de este proyecto escrito para datos tabulares, sin copiar modelos de imágenes ni sus métricas.
- Documentación de despliegue: [Next.js en Vercel](https://vercel.com/docs/frameworks/full-stack/nextjs). La consigna entregada menciona GitHub Pages; el usuario eligió Vercel y la referencia project-web contempla esa plataforma. Conviene confirmar con la cátedra que la modalidad de entrega admite esa sustitución.

El portal presenta resultados agregados y ejemplos retrospectivos. Antes de publicación definitiva debe resolverse la redistribución de las observaciones usadas en ejemplos y notebooks. La copia local completa permite revisar y reproducir el trabajo. No se publica un dataset crudo mediante descargas del portal.
