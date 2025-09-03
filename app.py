# app.py
# -------------------------------------------------------------------
# Lee Excel "estilo matriz" (sin encabezados explícitos), detecta:
#  - Problemática (Cadena de Resultados)
#  - Línea de acción (#)
#  - Acción (col 2), Indicador (col 3), Meta (col 4)
# Filtra por "Municipalidad" (o la palabra clave que indiques)
# Genera PDF con campos editables SOLO para "Resultado".
# Portada con imagen + colores institucionales.
# -------------------------------------------------------------------
import io
from typing import List, Dict, Optional

import streamlit as st
import pandas as pd

# PDF
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.lib.utils import ImageReader

# ====== PALETA INSTITUCIONAL ======
AZUL_OSCURO = colors.HexColor("#1F4E79")
AZUL_CLARO  = colors.HexColor("#DCEBF7")
BORDE       = colors.HexColor("#9BBBD9")
NEGRO       = colors.black

# Flag de campo multilínea en AcroForm (bit 13)
FF_MULTILINE = 4096

st.set_page_config(page_title="PDF editable – Seguimiento GL", layout="wide")
st.title("Generar PDF editable – Gobierno Local (Sembremos Seguridad)")

st.markdown(
    """
Cargá tu **Excel de la matriz** (como el que me compartiste). El sistema:
1) Reconstruye **Problemática → Línea de acción → Acción → Indicador → Meta** leyendo por **bloques**.
2) Filtra por líneas **municipales** (usando una palabra clave).
3) Genera un **PDF** con **solo “Resultado”** como campo editable (lo demás queda fijo).
"""
)

# ------------------ Helpers ------------------

def _s(x) -> str:
    return "" if pd.isna(x) else str(x).strip()

def parse_matriz(df: pd.DataFrame) -> pd.DataFrame:
    """
    Interpreta la hoja con columnas Unnamed:1.. y filas con contenido tipo:
      - "Cadena de Resultados: ...." en col1
      - "Línea de acción #..." en col1
      - Encabezados de sección como 'Actividades', 'Productos/Servicios', 'Efectos'
      - Filas de contenido en col2 (acción), col3 (indicador), col4 (meta)
    Devuelve un DataFrame con columnas: problematica, linea_accion, accion_estrategica, indicador, meta
    """
    # nombre columnas seguro por índice
    c1, c2, c3, c4 = 1, 2, 3, 4   # Unnamed:1..4
    cur_problem = ""
    cur_linea   = ""
    registros: List[Dict] = []

    for _, r in df.iterrows():
        t1 = _s(r.iloc[c1]) if c1 < len(r) else ""
        t2 = _s(r.iloc[c2]) if c2 < len(r) else ""
        t3 = _s(r.iloc[c3]) if c3 < len(r) else ""
        t4 = _s(r.iloc[c4]) if c4 < len(r) else ""

        # Detecta Cadena de Resultados
        if t1.lower().startswith("cadena de resultados"):
            # Extrae el texto luego de ': ' si existe
            cur_problem = t1.split(":", 1)[-1].strip() if ":" in t1 else t1
            continue

        # Detecta Línea de acción
        if t1.lower().startswith("línea de acción") or t1.lower().startswith("linea de accion"):
            cur_linea = t1.strip()
            continue

        # Ignora filas de encabezados de sub-sección
        if t1.lower() in {"causas", "actividades", "productos/servicios", "efectos", "impactos"}:
            continue

        # Filas de contenido: si hay algo en col2/col3/col4, se toma como (acción, indicador, meta)
        # (Algunas matrices traen texto solo en t2; igual se captura)
        hay_contenido = any([t2, t3, t4])
        if hay_contenido and (cur_problem or cur_linea):
            registros.append({
                "problematica": cur_problem,
                "linea_accion": cur_linea,
                "accion_estrategica": t2,
                "indicador": t3,
                "meta": t4,
            })

    return pd.DataFrame(registros)


def add_cover(c: canvas.Canvas, image_bytes: Optional[bytes]):
    y_top = A4[1] - 2.5*cm
    if image_bytes:
        try:
            img = ImageReader(io.BytesIO(image_bytes))
            iw, ih = img.getSize()
            max_w = A4[0] - 4*cm
            max_h = 7*cm
            scale = min(max_w/iw, max_h/ih)
            w, h = iw*scale, ih*scale
            x = (A4[0] - w)/2
            y = y_top - h
            c.drawImage(img, x, y, width=w, height=h, preserveAspectRatio=True, mask='auto')
            y_top = y - 0.8*cm
        except Exception:
            pass

    c.setFillColor(AZUL_OSCURO)
    c.setFont("Helvetica-Bold", 28)
    c.drawCentredString(A4[0]/2, y_top-1.6*cm, "INFORME DE SEGUIMIENTO")
    c.setFont("Helvetica-Bold", 22)
    c.drawCentredString(A4[0]/2, y_top-3.1*cm, "Gobierno Local")
    c.setFont("Helvetica", 11)
    c.setFillColor(NEGRO)
    c.drawCentredString(A4[0]/2, y_top-4.6*cm, "Sembremos Seguridad")
    c.showPage()


def draw_header(c: canvas.Canvas, page_num: int, total_pages: int):
    c.setFillColor(AZUL_CLARO)
    c.rect(1*cm, A4[1]-2.6*cm, A4[0]-2*cm, 1.6*cm, fill=1, stroke=0)
    c.setFillColor(AZUL_OSCURO)
    c.setFont("Helvetica-Bold", 14)
    c.drawString(1.2*cm, A4[1]-1.8*cm, "Informe de Seguimiento – Gobierno Local | Sembremos Seguridad")
    c.setFont("Helvetica", 9)
    c.drawRightString(A4[0]-1.2*cm, A4[1]-1.3*cm, f"Página {page_num} de {total_pages}")

def draw_footer(c: canvas.Canvas):
    c.setFillColor(colors.grey)
    c.setLineWidth(0.5)
    c.line(1*cm, 1.8*cm, A4[0]-1*cm, 1.8*cm)
    c.setFillColor(NEGRO)
    c.setFont("Helvetica", 8)
    c.drawString(1.2*cm, 1.3*cm, "Evidencias deben agregarse en la carpeta compartida designada.")

def wrap_text(c: canvas.Canvas, text: str, x: float, y: float, w: float, font="Helvetica", size=10) -> float:
    from reportlab.pdfbase.pdfmetrics import stringWidth
    words = (text or "").split()
    line = ""
    lh = 0.42*cm
    c.setFont(font, size)
    ty = y
    for word in words:
        test = (line + " " + word).strip()
        if stringWidth(test, font, size) <= w:
            line = test
        else:
            c.drawString(x, ty, line)
            ty -= lh
            line = word
    if line:
        c.drawString(x, ty, line)
        ty -= lh
    return ty

def section(c: canvas.Canvas, x: float, y: float, w: float, title: str, body: str) -> float:
    # Cabecera
    c.setFillColor(AZUL_CLARO)
    c.rect(x, y-0.9*cm, w, 0.9*cm, fill=1, stroke=0)
    c.setFillColor(AZUL_OSCURO)
    c.setFont("Helvetica-Bold", 11)
    c.drawString(x+0.2*cm, y-0.6*cm, title)
    # Cuerpo
    c.setFillColor(NEGRO)
    return wrap_text(c, body, x+0.2*cm, y-1.4*cm, w-0.4*cm)

def ensure_space(c: canvas.Canvas, y: float, need: float, page: int, total: int) -> (float, int):
    if y - need < 2.8*cm:
        draw_footer(c)
        c.showPage()
        draw_header(c, page+1, total)
        return A4[1] - 3.6*cm, page+1
    return y, page

def crear_pdf(regs: pd.DataFrame, header_img_bytes: Optional[bytes], filtro_txt: str) -> bytes:
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)

    # Portada
    add_cover(c, header_img_bytes)

    # Páginas de contenido
    total_pages_est = 1 + max(1, len(regs))  # estimación simple
    page = 1
    draw_header(c, page, total_pages_est)
    y = A4[1] - 3.6*cm
    x = 1.4*cm
    w = A4[0] - 2.8*cm

    idx = 1
    for _, r in regs.iterrows():
        prob = r["problematica"]
        lin  = r["linea_accion"]
        acc  = r["accion_estrategica"]
        ind  = r["indicador"]
        meta = r["meta"]

        # Problemática
        y, page = ensure_space(c, y, 3.0*cm, page, total_pages_est)
        y = section(c, x, y, w, "Cadena de Resultados / Problemática", prob)

        # Línea
        y, page = ensure_space(c, y, 2.6*cm, page, total_pages_est)
        y = section(c, x, y, w, "Línea de acción", lin)

        # Acción
        y, page = ensure_space(c, y, 2.6*cm, page, total_pages_est)
        y = section(c, x, y, w, "Acción Estratégica", acc)

        # Indicador
        y, page = ensure_space(c, y, 2.6*cm, page, total_pages_est)
        y = section(c, x, y, w, "Indicador", ind)

        # Meta
        y, page = ensure_space(c, y, 2.6*cm, page, total_pages_est)
        y = section(c, x, y, w, "Meta", meta)

        # Resultado (único editable)
        alto = 4.5*cm
        y, page = ensure_space(c, y, alto + 1.6*cm, page, total_pages_est)
        c.setFillColor(AZUL_CLARO)
        c.rect(x, y-0.9*cm, w, 0.9*cm, fill=1, stroke=0)
        c.setFillColor(AZUL_OSCURO)
        c.setFont("Helvetica-Bold", 11)
        c.drawString(x+0.2*cm, y-0.6*cm, "Resultado (rellenable por Gobierno Local)")
        c.setStrokeColor(BORDE)
        c.rect(x, y-(alto+1.2*cm), w, alto, fill=0, stroke=1)
        c.acroForm.textfield(
            name=f"resultado_{idx}",
            tooltip=f"Resultado línea {idx}",
            x=x+0.15*cm, y=y-(alto+1.2*cm)+0.15*cm, width=w-0.3*cm, height=alto-0.3*cm,
            borderStyle="inset", borderWidth=1, forceBorder=True,
            fontName="Helvetica", fontSize=10, fieldFlags=FF_MULTILINE
        )
        y -= (alto + 1.6*cm)
        idx += 1

    draw_footer(c)
    c.save()
    buf.seek(0)
    return buf.read()

# ------------------ UI ------------------

col1, col2 = st.columns([2,1])
with col1:
    excel_file = st.file_uploader("Subí tu Excel de la matriz", type=["xlsx", "xls"])
with col2:
    img_file = st.file_uploader("Imagen de portada (opcional)", type=["png", "jpg", "jpeg"])
    filtro_palabra = st.text_input("Filtro para líneas municipales", "municip")

st.divider()

if excel_file is None:
    st.info("Cargá el Excel para comenzar.")
    st.stop()

# Lee la primera hoja
try:
    df_raw = pd.read_excel(excel_file, engine="openpyxl")
except Exception as e:
    st.error(f"No se pudo leer el Excel: {e}")
    st.stop()

# Parseo por bloques (sin depender de encabezados)
regs = parse_matriz(df_raw)

if regs.empty:
    st.warning("No se detectaron bloques de datos (revisá que exista 'Cadena de Resultados' y 'Línea de acción #').")
    st.stop()

# Filtro municipal: aplica sobre todas las columnas concatenadas
if filtro_palabra.strip():
    kw = filtro_palabra.strip().lower()
    mask = regs.apply(lambda r: kw in (" ".join([str(x) for x in r.values])).lower(), axis=1)
    regs_filtrado = regs[mask]
else:
    regs_filtrado = regs

st.subheader("Vista previa (líneas incluidas)")
st.dataframe(regs_filtrado, use_container_width=True)

# Generar PDF
if st.button("Generar PDF editable"):
    img_bytes = img_file.read() if img_file is not None else None
    pdf_bytes = crear_pdf(regs_filtrado, img_bytes, filtro_palabra)
    st.success("PDF generado.")
    st.download_button(
        label="⬇️ Descargar PDF",
        data=pdf_bytes,
        file_name="Informe_Seguimiento_GobiernoLocal.pdf",
        mime="application/pdf"
    )









