# app.py
# -------------------------------------------------------------
# Lee Excel tipo matriz (sin header), detecta la fila de tabla:
#  Acciones Estrategicas | Indicador | Meta | Líder Estratégico | Co-gestores
# Reconstruye: Problemática y Línea de acción (arriba de la tabla).
# Filtra SOLO filas con Líder Estratégico = Municipalidad (coincidencia flexible).
# Genera PDF con colores institucionales y portada con imagen local del repo.
# Único campo editable en el PDF: "Resultado".
# -------------------------------------------------------------
import io
import re
import unicodedata
from typing import List, Dict, Optional, Tuple

import streamlit as st
import pandas as pd

# PDF
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.lib.utils import ImageReader

# ===== Paleta institucional =====
AZUL_OSCURO = colors.HexColor("#1F4E79")
AZUL_CLARO  = colors.HexColor("#DCEBF7")
BORDE       = colors.HexColor("#9BBBD9")
NEGRO       = colors.black
FF_MULTILINE = 4096  # AcroForm bit flag para campos multilínea

st.set_page_config(page_title="PDF editable – Seguimiento GL", layout="wide")
st.title("Generar PDF editable – Gobierno Local (Sembremos Seguridad)")

st.markdown(
    """
Carga tu **Excel de la matriz**. El sistema:
1) Ubica la **tabla** (fila con *Acciones Estrategicas / Indicador / Meta / Líder Estratégico / Co-gestores*).  
2) Reconstruye **Problemática** (*Cadena de Resultados: ...*) y **Línea de acción #...** desde las filas superiores.  
3) Toma **Acción**, **Indicador**, **Meta** y **Líder** de la tabla.  
4) Incluye **solo filas municipales** y genera un **PDF** con **“Resultado”** como **único campo editable**.
"""
)

# ---------------- Utils ----------------
def _norm(s: str) -> str:
    if s is None:
        return ""
    s = str(s).strip().lower()
    # quitar acentos para comparar de forma robusta
    s = "".join(ch for ch in unicodedata.normalize("NFKD", s) if not unicodedata.combining(ch))
    return s

def buscar_fila_encabezados(df: pd.DataFrame) -> Tuple[int, Dict[str,int]]:
    """
    Busca la fila de encabezados de la tabla y devuelve:
    - idx_fila
    - dict con índices de columnas {'accion': cA, 'indicador': cI, 'meta': cM, 'lider': cL, 'cogestores': cC?}
    """
    nrows, ncols = df.shape
    for i in range(nrows):
        row = [str(df.iat[i, j]) if pd.notna(df.iat[i, j]) else "" for j in range(ncols)]
        low = [_norm(x) for x in row]
        # claves mínimas: accion / indicador / meta (en cualquier posición de la fila)
        if any("acciones estrategicas" in x or "accion" in x for x in low) and \
           any("indicador" in x for x in low) and \
           any(re.search(r"\bmeta(s)?\b", x) for x in low):
            # mapear posiciones
            mapping = {"accion": None, "indicador": None, "meta": None, "lider": None, "cogestores": None}
            for j, cell in enumerate(low):
                if mapping["accion"] is None and ("acciones estrategicas" in cell or "accion" in cell):
                    mapping["accion"] = j
                if mapping["indicador"] is None and "indicador" in cell:
                    mapping["indicador"] = j
                if mapping["meta"] is None and re.search(r"\bmeta(s)?\b", cell):
                    mapping["meta"] = j
                if mapping["lider"] is None and ("lider estrategico" in cell or "líder estratégico" in cell):
                    mapping["lider"] = j
                if mapping["cogestores"] is None and ("co-gestores" in cell or "cogestores" in cell):
                    mapping["cogestores"] = j
            # validar mínimos
            if all(mapping[k] is not None for k in ["accion","indicador","meta"]):
                return i, mapping
    return -1, {}

def extraer_contexto_superior(df: pd.DataFrame, hasta_fila: int) -> Dict[str, str]:
    """
    Busca hacia arriba (desde 0 .. hasta_fila-1) los textos:
    - 'Cadena de Resultados: ...' -> problematica
    - 'Línea de acción #...'      -> linea_accion (la última encontrada antes de la tabla)
    Si hay varias líneas de acción por bloque, se actualizará durante el barrido de las filas de datos.
    """
    problem = ""
    linea = ""
    for i in range(hasta_fila):
        for j in range(df.shape[1]):
            val = df.iat[i, j]
            if pd.isna(val):
                continue
            txt = str(val).strip()
            low = _norm(txt)
            if low.startswith("cadena de resultados"):
                problem = txt.split(":", 1)[-1].strip() if ":" in txt else txt
            if low.startswith("linea de accion") or low.startswith("línea de acción"):
                linea = txt
    return {"problematica": problem, "linea_accion": linea}

def lider_es_muni(texto: str) -> bool:
    t = _norm(texto)
    return any(p in t for p in ["municip", "gobierno local", "alcald", "ayuntamiento"])

def parsear_matriz(df: pd.DataFrame) -> pd.DataFrame:
    """
    1) Localiza encabezados de tabla.
    2) Reconstruye problemática/ línea de acción del bloque superior.
    3) Recorre filas de datos; si en la misma fila hay nueva 'Línea de acción', la actualiza.
    """
    nrows, ncols = df.shape
    # Trabajar como strings
    S = df.astype(str).where(~df.isna(), "")

    fila_hdr, cols = buscar_fila_encabezados(S)
    if fila_hdr < 0:
        return pd.DataFrame(columns=["problematica","linea_accion","accion_estrategica","indicador","meta","lider"])

    ctx = extraer_contexto_superior(S, fila_hdr)
    cur_problem = ctx["problematica"]
    cur_linea = ctx["linea_accion"]

    registros: List[Dict] = []
    for i in range(fila_hdr + 1, nrows):
        row_vals = [S.iat[i, j] for j in range(ncols)]
        # si la fila contiene una nueva "Línea de acción", actualizar contexto
        for j in range(ncols):
            low = _norm(row_vals[j])
            if low.startswith("linea de accion") or low.startswith("línea de acción"):
                cur_linea = row_vals[j].strip()
        # si la fila contiene una nueva "Cadena de Resultados", actualizar problemática
        for j in range(ncols):
            low = _norm(row_vals[j])
            if low.startswith("cadena de resultados"):
                txt = row_vals[j].strip()
                cur_problem = txt.split(":",1)[-1].strip() if ":" in txt else txt

        # tomar valores por columnas mapeadas
        def _get(col_idx: Optional[int]) -> str:
            return row_vals[col_idx].strip() if (col_idx is not None and col_idx < ncols) else ""

        accion = _get(cols.get("accion"))
        indicador = _get(cols.get("indicador"))
        meta = _get(cols.get("meta"))
        lider = _get(cols.get("lider"))

        # cortar si fila vacía en las tres columnas clave
        if not any([accion, indicador, meta]):
            continue

        registros.append({
            "problematica": cur_problem,
            "linea_accion": cur_linea,
            "accion_estrategica": accion,
            "indicador": indicador,
            "meta": meta,
            "lider": lider
        })

    df_out = pd.DataFrame(registros)
    if not df_out.empty:
        # Limpieza básica
        df_out = df_out.replace({"None": "", "nan": ""})
        # Mantener solo filas municipales
        df_out = df_out[df_out["lider"].apply(lider_es_muni)]
    return df_out.reset_index(drop=True)

# ---------------- PDF ----------------
def portada_con_imagen(c: canvas.Canvas, image_path: Optional[str]):
    y_top = A4[1] - 2.5*cm
    if image_path:
        try:
            img = ImageReader(image_path)
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
    c.setFillColor(AZUL_CLARO)
    c.rect(x, y-0.9*cm, w, 0.9*cm, fill=1, stroke=0)
    c.setFillColor(AZUL_OSCURO)
    c.setFont("Helvetica-Bold", 11)
    c.drawString(x+0.2*cm, y-0.6*cm, title)
    c.setFillColor(NEGRO)
    return wrap_text(c, body, x+0.2*cm, y-1.4*cm, w-0.4*cm)

def ensure_space(c: canvas.Canvas, y: float, need: float, page: int, total: int) -> Tuple[float, int]:
    if y - need < 2.8*cm:
        draw_footer(c)
        c.showPage()
        draw_header(c, page+1, total)
        return A4[1] - 3.6*cm, page+1
    return y, page

def crear_pdf(df_rows: pd.DataFrame, image_path: Optional[str]) -> bytes:
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)
    # Portada
    portada_con_imagen(c, image_path)
    # Contenido
    total_pages_est = 1 + max(1, len(df_rows))
    page = 1
    draw_header(c, page, total_pages_est)
    y = A4[1] - 3.6*cm
    x = 1.4*cm
    w = A4[0] - 2.8*cm
    idx = 1

    for _, r in df_rows.iterrows():
        prob = r.get("problematica", "")
        lin  = r.get("linea_accion", "")
        acc  = r.get("accion_estrategica", "")
        ind  = r.get("indicador", "")
        meta = r.get("meta", "")

        y, page = ensure_space(c, y, 3.0*cm, page, total_pages_est)
        y = section(c, x, y, w, "Cadena de Resultados / Problemática", prob)

        y, page = ensure_space(c, y, 2.6*cm, page, total_pages_est)
        y = section(c, x, y, w, "Línea de acción", lin)

        y, page = ensure_space(c, y, 2.6*cm, page, total_pages_est)
        y = section(c, x, y, w, "Acción Estratégica", acc)

        y, page = ensure_space(c, y, 2.6*cm, page, total_pages_est)
        y = section(c, x, y, w, "Indicador", ind)

        y, page = ensure_space(c, y, 2.6*cm, page, total_pages_est)
        y = section(c, x, y, w, "Meta", meta)

        # Resultado (único campo editable)
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

# ---------------- UI ----------------
with st.sidebar:
    # Ruta de la imagen que ya tienes en tu repo
    ruta_img = st.text_input("Ruta imagen portada (en tu repo)", value="assets/encabezado.png")

excel_file = st.file_uploader("Subí tu Excel de la matriz", type=["xlsx", "xls"])

if excel_file is None:
    st.info("Cargá el Excel para continuar. La imagen de portada se lee desde la ruta local indicada en la barra lateral.")
    st.stop()

# Leer sin encabezados (tal cual tu archivo)
try:
    df_raw = pd.read_excel(excel_file, engine="openpyxl", header=None)
except Exception as e:
    st.error(f"No se pudo leer el Excel: {e}")
    st.stop()

# Parsear y filtrar municipalidad
regs = parsear_matriz(df_raw)

if regs.empty:
    st.warning("No se encontraron filas municipales o no se detectó la tabla. Revisá que existan los encabezados: Acciones / Indicador / Meta / Líder Estratégico.")
else:
    st.subheader("Vista previa (líneas incluidas)")
    st.dataframe(regs[["problematica","linea_accion","accion_estrategica","indicador","meta","lider"]], use_container_width=True)

    if st.button("Generar PDF editable"):
        pdf_bytes = crear_pdf(regs, ruta_img if ruta_img else None)
        st.success("PDF generado.")
        st.download_button(
            label="⬇️ Descargar PDF",
            data=pdf_bytes,
            file_name="Informe_Seguimiento_GobiernoLocal.pdf",
            mime="application/pdf"
        )








