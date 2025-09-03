# app.py
# -------------------------------------------------------
# Streamlit -> Cargar Excel -> Generar PDF con ACROFORMS
# Partes fijas (no editables): Problemática, Línea de acción, Acción, Indicador, Meta
# Único campo editable: Resultado (solo para filas con Lider Estratégico = Municipalidad)
# Colores institucionales + carátula con imagen
# -------------------------------------------------------
import io
from typing import Dict, List, Any, Optional

import streamlit as st
import pandas as pd

# PDF
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.lib import colors

# === PALETA (institucional) ===
AZUL_OSCURO = colors.HexColor("#1F4E79")
AZUL_CLARO  = colors.HexColor("#DCEBF7")
BORDE       = colors.HexColor("#9BBBD9")
NEGRO       = colors.black

# Flag de campo multilínea en AcroForm (bit 13)
FF_MULTILINE = 4096

st.set_page_config(page_title="Generador PDF – Informe Seguimiento", layout="wide")
st.title("Generar PDF editable – Gobierno Local (Sembremos Seguridad)")

st.markdown(
    """
Subí tu **Excel** con las líneas de acción. La aplicación:
1) Detecta columnas clave (nombres flexibles).  
2) Filtra **solo** las líneas donde el **Líder Estratégico** es *Municipalidad* (también acepta variantes como *Gobierno Local*, *Municipal*, etc.).  
3) Genera un **PDF** por todas las líneas encontradas, con **solo “Resultado”** como campo editable.
"""
)

# -------- Helpers de encabezados ----------

# Posibles nombres de columnas (normalización flexible)
CANDIDATAS = {
    "problematica": [
        "cadena de resultados", "problematica", "problemática", "nombre de la problemática",
        "categoria", "título cadena", "titulo cadena"
    ],
    "linea_accion": [
        "línea de acción", "linea de accion", "linea_de_accion", "linea", "línea"
    ],
    "accion_estrategica": [
        "acción estratégica", "accion estrategica", "accion_estrategica", "acciones estrategicas",
        "accion", "acción"
    ],
    "indicador": [
        "indicador", "indicadores", "nombre del indicador"
    ],
    "meta": [
        "meta", "metas", "meta cuantitativa", "meta_cuantitativa"
    ],
    "lider": [
        "líder estratégico", "lider estrategico", "lider_estrategico", "líder", "lider", "responsable"
    ],
    "id": [
        "n°", "n", "id", "numero", "número", "#"
    ]
}

def _norm(s: str) -> str:
    return (s or "").strip().lower().replace("\n"," ").replace("\r"," ")

def mapear_columnas(cols: List[str]) -> Dict[str, str]:
    """Devuelve un mapeo {'clave': 'nombre_col_en_df'} usando coincidencia flexible."""
    mapping = {}
    low = {_norm(c): c for c in cols}
    for clave, opciones in CANDIDATAS.items():
        for op in opciones:
            k = _norm(op)
            # buscar por contención
            for low_name, real_name in low.items():
                if k == low_name or k in low_name or low_name in k:
                    mapping[clave] = real_name
                    break
            if clave in mapping:
                break
    return mapping

def lider_es_muni(text: Any) -> bool:
    s = _norm(str(text))
    # Reglas amplias para municipalidad
    patrones = ["municip", "gobierno local", "alcald", "ayuntamiento"]
    return any(p in s for p in patrones)

# --------- Generación del PDF --------------

def draw_header(c: canvas.Canvas, page_num: int, total_pages: int, titulo:str, subtitulo:str):
    c.setFillColor(AZUL_CLARO)
    c.rect(1*cm, A4[1]-2.6*cm, A4[0]-2*cm, 1.6*cm, fill=1, stroke=0)
    c.setFillColor(AZUL_OSCURO)
    c.setFont("Helvetica-Bold", 14)
    c.drawString(1.2*cm, A4[1]-1.8*cm, titulo)
    c.setFont("Helvetica", 10)
    c.drawRightString(A4[0]-1.2*cm, A4[1]-1.3*cm, f"Página {page_num} de {total_pages}")
    if subtitulo:
        c.setFillColor(NEGRO)
        c.setFont("Helvetica", 9)
        c.drawString(1.2*cm, A4[1]-3.2*cm, subtitulo)

def draw_footer(c: canvas.Canvas):
    c.setFillColor(colors.grey)
    c.setLineWidth(0.5)
    c.line(1*cm, 1.8*cm, A4[0]-1*cm, 1.8*cm)
    c.setFillColor(NEGRO)
    c.setFont("Helvetica", 8)
    c.drawString(1.2*cm, 1.3*cm, "Evidencias deben agregarse en la carpeta compartida designada.")

def add_cover(c: canvas.Canvas, image_bytes: Optional[bytes]):
    # Portada con logos + título/subtítulo
    y_top = A4[1] - 2.5*cm
    if image_bytes:
        try:
            from reportlab.lib.utils import ImageReader
            img = ImageReader(io.BytesIO(image_bytes))
            # ancho máximo ~18cm, alto proporcional
            max_w = A4[0] - 4*cm
            iw, ih = img.getSize()
            scale = min(max_w/iw, 7*cm/ih)
            w = iw*scale; h = ih*scale
            x = (A4[0] - w)/2
            y = y_top - h - 1*cm
            c.drawImage(img, x, y, width=w, height=h, preserveAspectRatio=True, mask='auto')
            y_top = y - 0.5*cm
        except Exception:
            pass

    c.setFillColor(AZUL_OSCURO)
    c.setFont("Helvetica-Bold", 28)
    c.drawCentredString(A4[0]/2, y_top-2.0*cm, "INFORME DE SEGUIMIENTO")
    c.setFont("Helvetica-Bold", 22)
    c.setFillColor(AZUL_OSCURO)
    c.drawCentredString(A4[0]/2, y_top-3.5*cm, "Gobierno Local")
    c.setFont("Helvetica", 11)
    c.setFillColor(NEGRO)
    c.drawCentredString(A4[0]/2, y_top-5.0*cm, "Sembremos Seguridad")
    c.showPage()

def ensure_space(c: canvas.Canvas, cur_y: float, needed: float, page_num: int, total_pages: int):
    if cur_y - needed < 2.8*cm:
        draw_footer(c)
        c.showPage()
        draw_header(c, page_num + 1, total_pages, "Informe de Seguimiento – Gobierno Local | Sembremos Seguridad", "")
        return A4[1] - 3.6*cm, page_num + 1
    return cur_y, page_num

def section_block(c: canvas.Canvas, x: float, y: float, w: float, title: str, body: str, line_height: float=0.6*cm) -> float:
    # Cabecera azul
    c.setFillColor(AZUL_CLARO)
    c.rect(x, y-0.9*cm, w, 0.9*cm, fill=1, stroke=0)
    c.setFillColor(AZUL_OSCURO)
    c.setFont("Helvetica-Bold", 11)
    c.drawString(x+0.2*cm, y-0.6*cm, title)
    # Cuerpo
    c.setFillColor(NEGRO)
    c.setFont("Helvetica", 10)
    text_obj = c.beginText(x+0.2*cm, y-1.4*cm)
    wrap_w = w - 0.4*cm
    # Wrap manual simple
    for line in _wrap_text(body, wrap_w, c._fontname, 10):
        text_obj.textLine(line)
    c.drawText(text_obj)
    # Altura ocupada estimada
    lines = max(1, len(_wrap_text(body, wrap_w, c._fontname, 10)))
    return y - (1.4*cm + lines*0.42*cm + 0.4*cm)

def _wrap_text(text: str, max_width: float, font: str, size: int) -> List[str]:
    from reportlab.pdfbase.pdfmetrics import stringWidth
    words = str(text or "").split()
    lines, cur = [], ""
    for w in words:
        test = (cur + " " + w).strip()
        if stringWidth(test, font, size) <= max_width:
            cur = test
        else:
            lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    return lines or [""]

def crear_pdf(data: pd.DataFrame, m: Dict[str, str], header_img_bytes: Optional[bytes]) -> bytes:
    # Contar líneas municipales
    filas = data[lider_es_muni(data[m["lider"]])].copy() if "lider" in m else data.copy()
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)
    total_pages_est = 1 + max(1, len(filas))  # estimación simple
    # Portada
    add_cover(c, header_img_bytes)
    # Primera página de contenido
    page = 1
    draw_header(c, page, total_pages_est, "Informe de Seguimiento – Gobierno Local | Sembremos Seguridad", "")
    y = A4[1] - 3.6*cm
    x = 1.4*cm
    w = A4[0] - 2.8*cm

    index_global = 1
    for _, row in filas.iterrows():
        # Sección: Problemática (no editable)
        titulo_prob = "Cadena de Resultados / Problemática"
        texto_prob  = str(row.get(m.get("problematica",""), ""))

        y, page = ensure_space(c, y, 3.0*cm, page, total_pages_est)
        y = section_block(c, x, y, w, titulo_prob, texto_prob)

        # Sección: Línea de acción (no editable)
        titulo_lin = f"Línea de acción #{index_global}"
        texto_lin  = str(row.get(m.get("linea_accion",""), ""))
        y, page = ensure_space(c, y, 2.6*cm, page, total_pages_est)
        y = section_block(c, x, y, w, titulo_lin, texto_lin)

        # Acción estratégica (no editable)
        acc = str(row.get(m.get("accion_estrategica",""), ""))
        y, page = ensure_space(c, y, 2.6*cm, page, total_pages_est)
        y = section_block(c, x, y, w, "Acción Estratégica", acc)

        # Indicador (no editable)
        ind = str(row.get(m.get("indicador",""), ""))
        y, page = ensure_space(c, y, 2.6*cm, page, total_pages_est)
        y = section_block(c, x, y, w, "Indicador", ind)

        # Meta (no editable)
        meta = str(row.get(m.get("meta",""), ""))
        y, page = ensure_space(c, y, 2.6*cm, page, total_pages_est)
        y = section_block(c, x, y, w, "Meta", meta)

        # Resultado (ÚNICO editable – multilinea)
        alto = 4.5*cm
        y, page = ensure_space(c, y, alto + 1.6*cm, page, total_pages_est)
        # Cabecera para el campo editable
        c.setFillColor(AZUL_CLARO)
        c.rect(x, y-0.9*cm, w, 0.9*cm, fill=1, stroke=0)
        c.setFillColor(AZUL_OSCURO)
        c.setFont("Helvetica-Bold", 11)
        c.drawString(x+0.2*cm, y-0.6*cm, "Resultado (rellenable por Gobierno Local)")
        # Marco del campo
        c.setStrokeColor(BORDE)
        c.rect(x, y-(alto+1.2*cm), w, alto, fill=0, stroke=1)
        # Campo de formulario
        c.acroForm.textfield(
            name=f"resultado_{index_global}",
            tooltip=f"Resultado línea {index_global}",
            x=x+0.15*cm, y=y-(alto+1.2*cm)+0.15*cm, width=w-0.3*cm, height=alto-0.3*cm,
            borderStyle="inset", borderWidth=1, forceBorder=True,
            fontName="Helvetica", fontSize=10, fieldFlags=FF_MULTILINE
        )

        y -= (alto + 1.6*cm)
        index_global += 1

    draw_footer(c)
    c.save()
    buf.seek(0)
    return buf.read()

# ---------------- UI -----------------

col1, col2 = st.columns([2,1])
with col1:
    excel_file = st.file_uploader("Subí tu Excel", type=["xlsx", "xls"])
    st.caption("Acepta encabezados flexibles. Ejemplos: 'Línea de acción', 'Acción estratégica', 'Indicador', 'Meta', 'Líder estratégico'.")

with col2:
    img_file = st.file_uploader("Imagen de carátula (opcional)", type=["png", "jpg", "jpeg"])
    lock_lider = st.checkbox("Filtrar solo filas con Líder = Municipalidad", value=True)

st.divider()

if excel_file is not None:
    try:
        df = pd.read_excel(excel_file, engine="openpyxl")
    except Exception as e:
        st.error(f"No se pudo leer el Excel: {e}")
        st.stop()

    if df.empty:
        st.warning("El Excel no tiene filas.")
        st.stop()

    # Mapear columnas
    mapping = mapear_columnas(list(df.columns))
    faltantes = [k for k in ["linea_accion","accion_estrategica","indicador","meta","lider","problematica"] if k not in mapping]
    if faltantes:
        st.warning(
            "No se detectaron automáticamente estas columnas: "
            + ", ".join([f"**{k}**" for k in faltantes])
            + ". Renombra tus encabezados o añade columnas equivalentes."
        )

    # Vista previa + filtro
    preview = df.copy()
    if lock_lider and "lider" in mapping:
        preview = preview[df[mapping["lider"]].apply(lider_es_muni)]
    st.subheader("Vista previa de filas a incluir")
    st.dataframe(preview.head(50), use_container_width=True)

    # Generar PDF
    if st.button("Generar PDF editable"):
        img_bytes = img_file.read() if img_file is not None else None
        pdf_bytes = crear_pdf(preview, mapping, img_bytes)
        st.success("PDF generado.")
        st.download_button(
            label="⬇️ Descargar PDF",
            data=pdf_bytes,
            file_name="Informe_Seguimiento_GobiernoLocal.pdf",
            mime="application/pdf"
        )
else:
    st.info("Cargá un Excel para comenzar. Luego presioná **Generar PDF editable**.")








