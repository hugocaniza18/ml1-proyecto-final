import type { Metadata } from 'next';
import './globals.css';
export const metadata: Metadata = { title:'Roya de soja | FIUNA', description:'Comparación retrospectiva de modelos y ventanas meteorológicas de 7 y 14 días. Machine Learning 1, FIUNA.' };
export default function Layout({children}:{children:React.ReactNode}) {return <html lang="es"><body>{children}</body></html>}
