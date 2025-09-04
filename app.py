# app.py — Parser por contenido + multi-hoja + PDF agrupado con único campo editable ("Resultado")
import io, re, unicodedata
from typing import Dict, List, Optional, Tuple

import streamlit as st
import pandas as pd

# PDF
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.lib.utils import ImageReader

# ========= Estilo institucional =========
AZUL_OSCURO = colors.HexColor("#1F4E79")
AZUL_CLARO  = colors.HexColor("#DCEBF7")
BORDE       = colors.HexColor("#9BBBD9")
NEGRO       = colors.black
FF_MULTILINE = 4096  # Campo multilínea para AcroForm

# ========= Config Streamlit =========
st.set_page_config(page_title="PDF editable – Gobierno Local", layout="wide")
st.title("Generar PDF editable – Gobierno Local (Sembremos Seguridad)")

st.markdown(
    """
Subí tu **Excel** (una o varias hojas). La app detecta por **contenido**:
- **Cadena de Resultados**, **Línea de acción #**  
- Encabezados: **Acciones**, **Indicador** (*o* **Productos/Servicios**), **Meta** (*o* **Efectos**), **Líder Estratégico**, **Co-gestores**  
- Filtrado por **Municipalidad** (estricto o amplio)  
- El **PDF** muestra *Problemática* y *Línea* **una sola vez** y luego **fichas** por acción con **Resultado** como único campo editable.
"""
)

# ========= Utilidades =========
def _norm(s: str) -> str:
    """Normaliza texto: minúsculas sin acentos/espacios extremos."""
    if s is None:
        return ""
    s = str(s).strip().lower()
    return "".join(ch for ch in unicodedata.normalize("NFKD", s) if not unicodedata.combining(ch))

# Sinónimos que se ven en tus matrices
SIN_ACCION = ["acciones estrategicas", "acciones estrategica", "accion estrategica", "accion", "acciones"]
SIN_INDIC  = ["indicador", "indicadores", "productos/servicios", "producto/servicio", "producto", "servicio"]
SIN_META   = ["meta", "metas", "efecto", "efectos", "resultados esperados"]
SIN_LIDER  = ["lider estrategico", "líder estratégico", "lider", "responsable"]
SIN_COGE   = ["co-gestores", "cogestores", "co gestores", "co gestores:"]
SIN_CONSID = ["consideraciones", "observaciones", "comentarios"]

RE_CADENA = re.compile(r"^cadena\s+de\s+resultados", re.I)
RE_LINEA  = re.compile(r"^l[ií]nea\s+de\s+acci[oó]n\s*#?\s*\d*", re.I)

def _find_any(cell: str, keys: List[str]) -> bool:
    return any(k in _norm(cell) for k in keys)

def find_header_in_row(row_vals: List[str]) -> Dict[str, int]:
    """
    Dada una fila, intenta ubicar columnas por texto.
    Devuelve dict con indices: accion, indicador, meta, lider, cogestores, consideraciones (algunas pueden no existir).
    Para considerar "encabezado" deben existir al menos acción/indicador/meta.
    """
    n = len(row_vals)
    idx = {"accion": None, "indicador": None, "meta": None, "lider": None, "cogestores": None, "consideraciones": None}
    for j in range(n):
        v = row_vals[j]
        if idx["accion"] is None and _find_any(v, SIN_ACCION): idx["accion"] = j
        if idx["indicador"] is None and _find_any(v, SIN_INDIC): idx["indicador"] = j
        if idx["meta"] is None and _find_any(v, SIN_META): idx["meta"] = j
        if idx["lider"] is None and _find_any(v, SIN_LIDER): idx["lider"] = j
        if idx["cogestores"] is None and _find_any(v, SIN_COGE): idx["cogestores"] = j
        if idx["consideraciones"] is None and _find_any(v, SIN_CONSID): idx["consideraciones"] = j
    return idx if all(idx[k] is not None for k in ["accion","indicador","meta"]) else {}

def es_muni(texto: str) -> bool:
    t = _norm(texto)
    return any(p in t for p in ["municip", "gobierno local", "alcald", "ayuntamiento"])

# ========= Parser por hoja =========
def parse_sheet(df_raw: pd.DataFrame, sheet_name: str) -> pd.DataFrame:
    """
    Recorre toda la hoja:
      - Guarda última 'Cadena de Resultados' y 'Línea de acción #'
      - Detecta fila de encabezados (acepta sinónimos)
      - Extrae filas de datos hasta que se vacían las columnas clave
    Devuelve: problematica, linea_accion, accion_estrategica, indicador, meta, lider, cogestores, hoja
    """
    S = df_raw.astype(str).where(~df_raw.isna(), "")
    nrows, ncols = S.shape
    current_problem = ""
    current_linea   = ""
    rows: List[Dict] = []

    i = 0
    while i < nrows:
        row_vals = [S.iat[i, j].strip() for j in range(ncols)]

        # Actualizar contexto si en esta fila aparece una nueva Cadena/Línea
        for j in range(ncols):
            cell = row_vals[j]
            if RE_CADENA.match(cell):
                current_problem = cell.split(":",1)[-1].strip() if ":" in cell else cell.strip()
            if RE_LINEA.match(cell):
                current_linea = cell.strip()

        # Detectar encabezado
        hdr = find_header_in_row(row_vals)
        if hdr:
            # Leer filas de datos hacia abajo
            i += 1
            while i < nrows:
                row_vals = [S.iat[i, j].strip() for j in range(ncols)]
                # Contexto puede volver a actualizarse más abajo
                for j in range(ncols):
                    cell = row_vals[j]
                    if RE_CADENA.match(cell):
                        current_problem = cell.split(":",1)[-1].strip() if ":" in cell else cell.strip()
                    if RE_LINEA.match(cell):
                        current_linea = cell.strip()

                def get(col):
                    return row_vals[col].strip() if (col is not None and col < ncols) else ""

                acc  = get(hdr.get("accion"))
                ind  = get(hdr.get("indicador"))
                meta = get(hdr.get("meta"))
                lid  = get(hdr.get("lider"))
                cog  = get(hdr.get("cogestores"))

                # fin del bloque si no hay datos en las 3 clave
                if not any([acc, ind, meta]):
                    break

                rows.append({
                    "problematica": current_problem,
                    "linea_accion": current_linea,
                    "accion_estrategica": acc,
                    "indicador": ind,
                    "meta": meta,
                    "lider": lid,
                    "cogestores": cog,
                    "hoja": sheet_name,
                })
                i += 1
            continue

        i += 1

    return pd.DataFrame(rows)

# ========= Parser de libro completo =========
def parse_workbook(xls_bytes) -> pd.DataFrame:
    xls = pd.ExcelFile(xls_bytes, engine="openpyxl")
    todo = []
    for sheet in xls.sheet_names:
        df = pd.read_excel(xls, sheet_name=sheet, header=None, engine="openpyxl")
        parsed = parse_sheet(df, sheet)
        if not parsed.empty:
            todo.append(parsed)
    if todo:
        return pd.concat(todo, ignore_index=True)
    return pd.DataFrame(columns=["problematica","linea_accion","accion_estrategica","indicador","meta","lider","cogestores","hoja"])

# ========= Helpers PDF (agrupado) =========
def portada(c: canvas.Canvas, image_path: Optional[str]):
    y_top = A4[1] - 2.5*cm
    if image_path:
        try:
            img = ImageReader(image_path)
            iw, ih = img.getSize()
            max_w, max_h = A4[0]-4*cm, 7*cm
            scale = min(max_w/iw, max_h/ih)
            w, h = iw*scale, ih*scale
            x = (A4[0]-w)/2; y = y_top - h
            c.drawImage(img, x, y, width=w, height=h, preserveAspectRatio=True, mask='auto')
            y_top = y - 0.8*cm
        except Exception:
            pass
    c.setFillColor(AZUL_OSCURO)
    c.setFont("Helvetica-Bold", 28); c.drawCentredString(A4[0]/2, y_top-1.6*cm, "INFORME DE SEGUIMIENTO")
    c.setFont("Helvetica-Bold", 22); c.drawCentredString(A4[0]/2, y_top-3.1*cm, "Gobierno Local")
    c.setFont("Helvetica", 11); c.setFillColor(NEGRO); c.drawCentredString(A4[0]/2, y_top-4.6*cm, "Sembremos Seguridad")
    c.showPage()

def header(c: canvas.Canvas, page: int, total: int):
    c.setFillColor(AZUL_CLARO); c.rect(1*cm, A4[1]-2.6*cm, A4[0]-2*cm, 1.6*cm, fill=1, stroke=0)
    c.setFillColor(AZUL_OSCURO); c.setFont("Helvetica-Bold", 14)
    c.drawString(1.2*cm, A4[1]-1.8*cm, "Informe de Seguimiento – Gobierno Local | Sembremos Seguridad")
    c.setFont("Helvetica", 9); c.drawRightString(A4[0]-1.2*cm, A4[1]-1.3*cm, f"Página {page} de {total}")

def footer(c: canvas.Canvas):
    c.setFillColor(colors.grey); c.setLineWidth(0.5)
    c.line(1*cm, 1.8*cm, A4[0]-1*cm, 1.8*cm)
    c.setFillColor(NEGRO); c.setFont("Helvetica", 8)
    c.drawString(1.2*cm, 1.3*cm, "Evidencias deben agregarse en la carpeta compartida designada.")

def wrap_text(c, text, x, y, w, font="Helvetica", size=10):
    from reportlab.pdfbase.pdfmetrics import stringWidth
    words = (text or "").split(); line=""; lh=0.42*cm; c.setFont(font, size); ty=y
    for wd in words:
        test=(line+" "+wd).strip()
        if stringWidth(test, font, size)<=w: line=test
        else: c.drawString(x,ty,line); ty-=lh; line=wd
    if line: c.drawString(x,ty,line); ty-=lh
    return ty

def section_bar(c, x, y, w, title):
    c.setFillColor(AZUL_CLARO); c.rect(x, y-0.9*cm, w, 0.9*cm, fill=1, stroke=0)
    c.setFillColor(AZUL_OSCURO); c.setFont("Helvetica-Bold", 11)
    c.drawString(x+0.2*cm, y-0.6*cm, title)
    return y-1.1*cm

def kv_item(c, x, y, w, label, value):
    c.setFillColor(NEGRO); c.setFont("Helvetica-Bold", 10)
    c.drawString(x, y, f"{label}:")
    return wrap_text(c, value, x+3.3*cm, y, w-3.5*cm, font="Helvetica", size=10)

def ensure_space(c, y, need, page, total):
    if y-need < 2.8*cm:
        footer(c); c.showPage(); header(c, page+1, total)
        return A4[1]-3.6*cm, page+1
    return y, page

def ficha_accion(c, x, y, w, idx, fila) -> float:
    """
    Tarjeta por Acción Estratégica con:
    - Acción Estratégica
    - Indicador
    - Meta
    - Líder Estratégico
    - Resultado (editable)
    """
    # Marco de ficha
    c.setStrokeColor(BORDE); c.setFillColor(colors.white)
    c.rect(x, y-5.8*cm, w, 5.8*cm, fill=0, stroke=1)

    # Encabezado de la ficha
    c.setFillColor(AZUL_CLARO); c.rect(x, y-0.9*cm, w, 0.9*cm, fill=1, stroke=0)
    c.setFillColor(AZUL_OSCURO); c.setFont("Helvetica-Bold", 11)
    c.drawString(x+0.2*cm, y-0.6*cm, f"Acción #{idx}")

    y_text = y-1.3*cm
    y_text = kv_item(c, x+0.2*cm, y_text, w-0.4*cm, "Acción Estratégica", fila.get("accion_estrategica",""))
    y_text -= 0.2*cm
    y_text = kv_item(c, x+0.2*cm, y_text, w-0.4*cm, "Indicador", fila.get("indicador",""))
    y_text -= 0.2*cm
    y_text = kv_item(c, x+0.2*cm, y_text, w-0.4*cm, "Meta", fila.get("meta",""))
    y_text -= 0.2*cm
    y_text = kv_item(c, x+0.2*cm, y_text, w-0.4*cm, "Líder Estratégico", fila.get("lider",""))

    # Resultado (único editable)
    alto = 2.2*cm
    c.setFillColor(AZUL_CLARO)
    c.rect(x+0.2*cm, y_text-0.7*cm, w-0.4*cm, 0.7*cm, fill=1, stroke=0)
    c.setFillColor(AZUL_OSCURO); c.setFont("Helvetica-Bold", 10)
    c.drawString(x+0.35*cm, y_text-0.45*cm, "Resultado (rellenable por Gobierno Local)")
    c.setStrokeColor(BORDE)
    c.rect(x+0.2*cm, y_text-(alto+1.0*cm), w-0.4*cm, alto, fill=0, stroke=1)
    c.acroForm.textfield(
        name=f"resultado_{idx}",
        tooltip=f"Resultado acción {idx}",
        x=x+0.25*cm, y=y_text-(alto+1.0*cm)+0.1*cm,
        width=w-0.5*cm, height=alto-0.2*cm,
        borderStyle="inset", borderWidth=1, forceBorder=True,
        fontName="Helvetica", fontSize=10, fieldFlags=FF_MULTILINE
    )
    return y_text-(alto+1.4*cm)

def build_pdf_grouped(rows: pd.DataFrame, image_path: Optional[str]) -> bytes:
    """
    Para cada (Problemática, Línea): imprime Problemática y Línea UNA sola vez
    y luego N fichas (una por acción) con Resultado editable.
    """
    buf = io.BytesIO(); c = canvas.Canvas(buf, pagesize=A4)
    portada(c, image_path)

    # Agrupar por (problematica, linea_accion)
    grupos = []
    if rows.empty:
        grupos = [(("", ""), rows)]
    else:
        for (prob, lin), g in rows.groupby(["problematica","linea_accion"], dropna=False):
            grupos.append(((prob or ""), (lin or ""), g.reset_index(drop=True)))

    total = 1 + max(1, len(grupos))
    page = 1; header(c, page, total)
    x = 1.4*cm; w = A4[0]-2.8*cm; y = A4[1]-3.6*cm

    for (prob, lin, gdf) in grupos:
        # Cabeceras del bloque (una sola vez)
        y, page = ensure_space(c, y, 4.0*cm, page, total)
        y = section_bar(c, x, y, w, "Cadena de Resultados / Problemática")
        y = wrap_text(c, prob, x+0.2*cm, y, w-0.4*cm)
        y -= 0.2*cm
        y = section_bar(c, x, y, w, "Línea de acción")
        y = wrap_text(c, lin, x+0.2*cm, y, w-0.4*cm)
        y -= 0.3*cm

        # Fichas de acciones
        for idx, fila in gdf.iterrows():
            y, page = ensure_space(c, y, 7.0*cm, page, total)
            y = ficha_accion(c, x, y, w, idx+1, fila)

        # Separación entre bloques
        y -= 0.6*cm
        if y < 4*cm:
            footer(c); c.showPage(); page += 1; header(c, page, total); y = A4[1]-3.6*cm

    footer(c); c.save(); buf.seek(0); return buf.read()

# ========= UI =========
with st.sidebar:
    ruta_img = st.text_input("Ruta imagen portada (en tu repo)", value="assets/encabezado.png")
    filtro = st.selectbox("Filtrar municipalidad", ["Líder contiene", "Líder o Co-gestores", "Sin filtro"])
    modo_hojas = st.radio("Hojas a procesar", ["Todas", "Elegir una"])

excel_file = st.file_uploader("Subí tu Excel (multipestaña o una sola)", type=["xlsx","xls"])
if not excel_file:
    st.info("Cargá el Excel para comenzar. La portada usa la imagen de la ruta indicada en la barra lateral.")
    st.stop()

# Procesar hojas
if modo_hojas == "Elegir una":
    xls_names = pd.ExcelFile(excel_file, engine="openpyxl").sheet_names
    hoja_sel = st.selectbox("Hoja a procesar", xls_names)
    df_hoja = pd.read_excel(excel_file, sheet_name=hoja_sel, header=None, engine="openpyxl")
    regs = parse_sheet(df_hoja, hoja_sel)
else:
    regs = parse_workbook(excel_file)

st.caption(f"Filas detectadas (todas las hojas seleccionadas): **{len(regs)}**")

# Aplicar filtro municipal
if filtro == "Líder contiene":
    regs_f = regs[regs["lider"].apply(es_muni)]
elif filtro == "Líder o Co-gestores":
    regs_f = regs[regs.apply(lambda r: es_muni(r.get("lider","")) or es_muni(r.get("cogestores","")), axis=1)]
else:
    regs_f = regs

st.caption(f"Filas después del filtro: **{len(regs_f)}**")
st.subheader("Vista previa")
cols = ["hoja","problematica","linea_accion","accion_estrategica","indicador","meta","lider","cogestores"]
st.dataframe((regs_f if not regs_f.empty else regs)[cols], use_container_width=True)

if st.button("Generar PDF editable"):
    data = regs_f if not regs_f.empty else regs
    pdf_bytes = build_pdf_grouped(data, ruta_img or None)
    st.success("PDF generado.")
    st.download_button("⬇️ Descargar PDF", data=pdf_bytes, file_name="Informe_Seguimiento_GobiernoLocal.pdf", mime="application/pdf")




