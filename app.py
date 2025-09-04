# app.py — parser por CONTENIDO + multi-hoja + PDF editable (solo “Resultado”)
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

# ===== Estilo institucional =====
AZUL_OSCURO = colors.HexColor("#1F4E79")
AZUL_CLARO  = colors.HexColor("#DCEBF7")
BORDE       = colors.HexColor("#9BBBD9")
NEGRO       = colors.black
FF_MULTILINE = 4096  # Campo multilínea para AcroForm

st.set_page_config(page_title="PDF editable – Seguimiento GL", layout="wide")
st.title("Generar PDF editable – Gobierno Local (Sembremos Seguridad)")

st.markdown(
    """
Subí tu **Excel**. El sistema busca por **contenido** (no por celdas fijas):

- *Cadena de Resultados: …*  
- *Línea de acción #…*  
- Encabezados: **Acciones** / **Indicador** (o *Productos/Servicios*) / (*Consideraciones* si existe) / **Meta** (o *Efectos*) / **Líder Estratégico** / **Co-gestores**  
- Filtrado por **Municipalidad** (opcional)  
- PDF: todo fijo salvo **“Resultado”** (editable)
"""
)

# ================= Utils =================
def _norm(s: str) -> str:
    if s is None: return ""
    s = str(s).strip().lower()
    # quitar acentos para comparar
    return "".join(ch for ch in unicodedata.normalize("NFKD", s) if not unicodedata.combining(ch))

# sinónimos que aparecen en tus plantillas
SIN_ACCION = ["acciones estrategicas", "acciones estrategica", "accion estrategica", "accion", "acciones"]
SIN_INDIC  = ["indicador", "indicadores", "productos/servicios", "producto/servicio", "producto", "servicio"]
SIN_META   = ["meta", "metas", "efecto", "efectos", "resultados esperados"]
SIN_LIDER  = ["lider estrategico", "líder estratégico", "lider", "responsable"]
SIN_COGE   = ["co-gestores", "cogestores", "co gestores", "co gestores:"]
SIN_CONSID = ["consideraciones", "observaciones", "comentarios"]

RE_CADENA = re.compile(r"^cadena\s+de\s+resultados", re.I)
RE_LINEA  = re.compile(r"^l[ií]nea\s+de\s+acci[oó]n\s*#?\s*\d*", re.I)

def _find_any(cell: str, keys: List[str]) -> bool:
    low = _norm(cell)
    return any(k in low for k in keys)

def find_header_in_row(row_vals: List[str]) -> Dict[str, int]:
    """
    Dada una fila, intenta ubicar columnas por texto.
    Devuelve dict con indices: accion, indicador, meta, lider, cogestores, consideraciones?
    Algunas pueden no existir (None).
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
    # al menos acción/indicador/meta deben existir para considerar "encabezado"
    if all(idx[k] is not None for k in ["accion","indicador","meta"]):
        return idx
    return {}

def es_muni(texto: str) -> bool:
    t = _norm(texto)
    return any(p in t for p in ["municip", "gobierno local", "alcald", "ayuntamiento"])

# ================ PARSER por hoja ================
def parse_sheet(df_raw: pd.DataFrame, sheet_name: str) -> pd.DataFrame:
    """
    Recorre TODAS las celdas:
      - guarda última 'Cadena de Resultados' y 'Línea de acción #'
      - detecta fila de encabezados (con sinónimos)
      - extrae filas de datos hasta que se vacíen las columnas clave
    Devuelve: problematica, linea_accion, accion_estrategica, indicador, meta, lider, cogestores, hoja
    """
    S = df_raw.astype(str).where(~df_raw.isna(), "")
    nrows, ncols = S.shape
    current_problem = ""
    current_linea   = ""
    header_idx: Optional[Dict[str,int]] = None
    rows: List[Dict] = []

    i = 0
    while i < nrows:
        row_vals = [S.iat[i, j].strip() for j in range(ncols)]

        # actualizar contexto (título y línea) en cualquier columna
        for j in range(ncols):
            cell = row_vals[j]
            if RE_CADENA.match(cell):
                # tomar texto luego de ':' si existe
                current_problem = cell.split(":",1)[-1].strip() if ":" in cell else cell.strip()
            if RE_LINEA.match(cell):
                current_linea = cell.strip()

        # detectar encabezado
        hdr = find_header_in_row(row_vals)
        if hdr:
            header_idx = hdr
            # leer filas de datos hacia abajo
            i += 1
            while i < nrows:
                row_vals = [S.iat[i, j].strip() for j in range(ncols)]
                # si aparece nueva cadena o nueva línea, actualizo contexto y NO rompo (hay matrices con varias tablas)
                for j in range(ncols):
                    cell = row_vals[j]
                    if RE_CADENA.match(cell):
                        current_problem = cell.split(":",1)[-1].strip() if ":" in cell else cell.strip()
                    if RE_LINEA.match(cell):
                        current_linea = cell.strip()

                def get(col):
                    return row_vals[col].strip() if (col is not None and col < ncols) else ""

                acc  = get(header_idx.get("accion"))
                ind  = get(header_idx.get("indicador"))
                meta = get(header_idx.get("meta"))
                lid  = get(header_idx.get("lider"))
                cog  = get(header_idx.get("cogestores"))

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
            # después de consumir el bloque, continuo sin perder el while principal
            continue

        i += 1

    return pd.DataFrame(rows)

# ================ PARSER de libro completo ================
def parse_workbook(xls_bytes) -> pd.DataFrame:
    """
    Lee todas las hojas y concatena resultados.
    """
    xls = pd.ExcelFile(xls_bytes, engine="openpyxl")
    todo = []
    for sheet in xls.sheet_names:
        df = pd.read_excel(xls, sheet_name=sheet, header=None, engine="openpyxl")
        parsed = parse_sheet(df, sheet)
        if not parsed.empty:
            todo.append(parsed)
    return pd.concat(todo, ignore_index=True) if todo else pd.DataFrame(
        columns=["problematica","linea_accion","accion_estrategica","indicador","meta","lider","cogestores","hoja"]
    )

# ================= PDF helpers =================
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

def section(c, x, y, w, title, body):
    c.setFillColor(AZUL_CLARO); c.rect(x, y-0.9*cm, w, 0.9*cm, fill=1, stroke=0)
    c.setFillColor(AZUL_OSCURO); c.setFont("Helvetica-Bold", 11); c.drawString(x+0.2*cm, y-0.6*cm, title)
    c.setFillColor(NEGRO); return wrap_text(c, body, x+0.2*cm, y-1.4*cm, w-0.4*cm)

def ensure_space(c, y, need, page, total):
    if y-need < 2.8*cm:
        footer(c); c.showPage(); header(c, page+1, total)
        return A4[1]-3.6*cm, page+1
    return y, page

def build_pdf(rows: pd.DataFrame, image_path: Optional[str]) -> bytes:
    buf = io.BytesIO(); c = canvas.Canvas(buf, pagesize=A4)
    portada(c, image_path)
    total = 1 + max(1, len(rows))
    page = 1; header(c, page, total)
    y = A4[1]-3.6*cm; x = 1.4*cm; w = A4[0]-2.8*cm; idx = 1

    for _, r in rows.iterrows():
        y, page = ensure_space(c, y, 3.0*cm, page, total); y = section(c, x, y, w, "Cadena de Resultados / Problemática", r.get("problematica",""))
        y, page = ensure_space(c, y, 2.6*cm, page, total); y = section(c, x, y, w, "Línea de acción", r.get("linea_accion",""))
        y, page = ensure_space(c, y, 2.6*cm, page, total); y = section(c, x, y, w, "Acción Estratégica", r.get("accion_estrategica",""))
        y, page = ensure_space(c, y, 2.6*cm, page, total); y = section(c, x, y, w, "Indicador", r.get("indicador",""))
        y, page = ensure_space(c, y, 2.6*cm, page, total); y = section(c, x, y, w, "Meta", r.get("meta",""))

        # único campo editable
        alto = 4.5*cm
        y, page = ensure_space(c, y, alto+1.6*cm, page, total)
        c.setFillColor(AZUL_CLARO); c.rect(x, y-0.9*cm, w, 0.9*cm, fill=1, stroke=0)
        c.setFillColor(AZUL_OSCURO); c.setFont("Helvetica-Bold", 11)
        c.drawString(x+0.2*cm, y-0.6*cm, "Resultado (rellenable por Gobierno Local)")
        c.setStrokeColor(BORDE); c.rect(x, y-(alto+1.2*cm), w, alto, fill=0, stroke=1)
        c.acroForm.textfield(
            name=f"resultado_{idx}", tooltip=f"Resultado línea {idx}",
            x=x+0.15*cm, y=y-(alto+1.2*cm)+0.15*cm, width=w-0.3*cm, height=alto-0.3*cm,
            borderStyle="inset", borderWidth=1, forceBorder=True,
            fontName="Helvetica", fontSize=10, fieldFlags=FF_MULTILINE
        )
        y -= (alto+1.6*cm); idx += 1

    footer(c); c.save(); buf.seek(0); return buf.read()

# ================= UI =================
with st.sidebar:
    ruta_img = st.text_input("Ruta imagen portada (del repo)", value="assets/encabezado.png")
    filtro = st.selectbox("Filtrar municipalidad", ["Líder contiene", "Líder o Co-gestores", "Sin filtro"])
    modo_hojas = st.radio("Hojas a procesar", ["Todas", "Elegir una"])

excel_file = st.file_uploader("Subí tu Excel (toma todas las hojas por defecto)", type=["xlsx","xls"])
if not excel_file:
    st.info("Cargá el Excel para comenzar. La portada usa la imagen de la ruta indicada en la barra lateral.")
    st.stop()

# Si eligen una hoja específica
if modo_hojas == "Elegir una":
    xls_names = pd.ExcelFile(excel_file, engine="openpyxl").sheet_names
    hoja_sel = st.selectbox("Hoja a procesar", xls_names)
    xls_bytes = excel_file
    df = pd.read_excel(xls_bytes, sheet_name=hoja_sel, header=None, engine="openpyxl")
    regs = parse_sheet(df, hoja_sel)
else:
    regs = parse_workbook(excel_file)

st.caption(f"Filas detectadas (todas las hojas seleccionadas): **{len(regs)}**")

# aplicar filtro municipal
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
    pdf_bytes = build_pdf(data, ruta_img or None)
    st.success("PDF generado.")
    st.download_button("⬇️ Descargar PDF", data=pdf_bytes, file_name="Informe_Seguimiento_GobiernoLocal.pdf", mime="application/pdf")




