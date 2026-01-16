# Scraper
import requests
from bs4 import BeautifulSoup
import pdfplumber
import pandas as pd
import os

def extract_tables_from_pdfs(pdf_folder="pdf_smn"):
    datos = []

    for archivo in os.listdir(pdf_folder):
        if archivo.endswith(".pdf"):
            ruta = os.path.join(pdf_folder, archivo)
            tipo = archivo.split("_")[0]
            año = archivo.split("_")[1].replace(".pdf", "")
            print(f"📄 Procesando {archivo}")

            with pdfplumber.open(ruta) as pdf:
                for page in pdf.pages:
                    table = page.extract_table()
                    if table:
                        df = pd.DataFrame(table[1:], columns=table[0])
                        df["año"] = año
                        df["tipo"] = tipo
                        datos.append(df)
                    break  # normalmente una sola tabla por PDF

    return pd.concat(datos, ignore_index=True)


def datos_climatologicos_mex():
    BASE_URL = "https://smn.conagua.gob.mx"
    IFRAME_URL = "https://smn.conagua.gob.mx/tools/GUI/Visor.php?id=57"

    # Obtener contenido del iframe
    res = requests.get(IFRAME_URL)
    soup = BeautifulSoup(res.content, 'html.parser')

    # Buscar el bloque que contiene el string de JSON embebido
    script = soup.find_all('script')

    pdf_links = set()

    for s in script:
        if "var json" in s.text and "tabla" in s.text:
            texto = s.string
            break

    # Extraer los <a href=...>.pdf desde el contenido del JSON embebido en JS
    soup_fake = BeautifulSoup(texto, "html.parser")
    for a in soup_fake.find_all("a"):
        href = a.get("href")
        if href and href.endswith(".pdf"):
            pdf_links.add(BASE_URL + href)

    # Crear carpeta
    os.makedirs("pdf_smn", exist_ok=True)

    # Descargar archivos
    for link in sorted(pdf_links):
        filename = link.split("/")[-2] + "_" + link.split("/")[-1]
        path = os.path.join("pdf_smn", filename)
        print(f"⬇️  Descargando {filename}...")
        r = requests.get(link)
        with open(path, "wb") as f:
            f.write(r.content)

    print("✅ Todos los PDFs han sido descargados.")

    df_resultado = extract_tables_from_pdfs()
    #df_resultado.to_csv("temperatura_lluvia_smn.csv", index=False)
    print("✅ Datos extraídos y guardados en 'temperatura_lluvia_smn.csv'")

    # Meses en orden
    meses = ['Ene','Feb','Mar','Abr','May','Jun','Jul','Ago','Sep','Oct','Nov','Dic']

    # Derretimos el DataFrame
    df_largo = df_resultado.melt(
        id_vars=['Estado', 'año', 'tipo'],
        value_vars=meses,
        var_name='mes',
        value_name='valor'
    )

    # Diccionario de meses a números
    mapa_meses = {
        'Ene': 1, 'Feb': 2, 'Mar': 3, 'Abr': 4,
        'May': 5, 'Jun': 6, 'Jul': 7, 'Ago': 8,
        'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dic': 12
    }

    # Convertimos 'mes' a número
    df_largo['mes_num'] = df_largo['mes'].map(mapa_meses)

    # Creamos columna de fecha con primer día del mes
    df_largo['fecha'] = pd.to_datetime(dict(year=df_largo['año'], month=df_largo['mes_num'], day=1))

    # Orden final
    df_largo = df_largo[['Estado', 'tipo', 'fecha', 'valor', 'mes_num']]

    # Resultado
    print(df_largo.head())
    df_largo['valor'] = pd.to_numeric(df_largo['valor'], errors='coerce')
    
    return df_largo
