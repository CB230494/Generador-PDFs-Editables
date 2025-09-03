# app.py
# -------------------------------------------------------------------
# Lee Excel "tipo matriz" sin depender de encabezados fijos.
# Encuentra dinámicamente: Problemática, Línea de acción, Acción, Indicador, Meta.
# Genera PDF con único campo editable: "Resultado".
# Usa una imagen local del repo para la portada (indicar ruta).
# -------------------------------------------------------------------
import io
import re
from typing import List, Dict, Optional, Tuple

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
Cargá tu **Excel de la matriz**. El sistema:
1) Reconstruye **Problemática → Línea de acción → Acción → Indicador → Meta** escaneando todas las celdas.
2) Detecta la fila de **encabezados** (*Acciones / Indicador / Meta*) aunque cambien de columna.
3) (Opcional) filtra por líneas **municipales** con una palabra clave.
4) Genera un **PDF** con **solo “Resultado”** como campo editable.
"""
)

# ------------------ Utils ------------------

def _s(x) -> str:
    return "" if pd.isna(x) else str(x).strip()

def find_header_cols(row_vals: List[str]) -> Tuple[Optional[int], Optional[int], Optional[int]]:
    """
    Dada una fila (lista de strings), intenta detectar las columnas de:
    Acciones (acción estratégica), Indicador, Meta.
    Devuelve índices (c_accion, c_indic, c_meta) o None si no encuentra.
    """
    c_acc = c_ind = c_meta = None
    for idx, val in enumerate(row_vals):
        v = val.lower()
        if c_acc is None and ("accion" in v or "acción" in v):
            c_acc = idx
        if c_ind is None and "indicador" in v:
            c_ind = idx
        if c_meta is None and re.search(r"\bmeta(s)?\b", v):
            c_meta = idx
    return c_acc, c_ind, c_meta

def parse_matriz_dynamic(df: pd.DataFrame) -> pd.DataFrame:
    """
    Parser dinámico:
      - Busca celdas con 'Cadena de Resultados' y 'Línea de acción'.
      - Detecta fila de encabezados (Acciones/Indicador/Meta) y sus columnas.
      - Captura filas de contenido hasta que cambie de bloque o se vacíe.
    Devuelve columnas: problematica, linea_accion, accion_estrategica, indicador, meta
    """
    nrows, ncols = df.shape
    # Normalizar a strings
    S = df.astype(str).where(~df.isna(), "")

    registros: List[Dict] = []
    cur_problem = ""
    cur_linea   = ""
    acc_c = ind_c = meta_c = None

    i = 0
    while i < nrows:
        row_vals = [_s(S.iat[i, j]) for j in range(ncols)]
        row_concat = " | ".join(row_vals).lower()

        # Detecta Problemática
        found_prob = None
        for j in range(ncols):
            v = row_vals[j]
            if v.lower().startswith("cadena de resultados"):
                # texto después de ':'
                found_prob = v.split(":", 1)[-1].strip() if ":" in v else v
                break
        if found_prob is not None:
            cur_problem = found_prob
            i += 1
            continue

        # Detecta Línea de acción
        found_linea = None
        for j in range(ncols):
            v = row_vals[j].lower()
            if v.startswith("línea de acción") or v.startswith("linea de accion"):
                found_linea = row_vals[j]
                break
        if found_linea is not None:
            cur_linea = found_linea
            # al cambiar de línea, “olvidar” mapeo de columnas para encontrarlas de nuevo si reaparecen más abajo
            acc_c = ind_c = meta_c = None
            i += 1
            continue

        # Detecta encabezados Accion/Indicador/Meta en esta fila
        if acc_c is None or ind_c is None or meta_c is None:
            a, b, c = find_header_cols(row_vals)
            if a is not None and b is not None and c is not None:
                acc_c, ind_c, meta_c = a, b, c
                i += 1
                continue

        # Si ya conocemos las columnas de contenido, leer filas de datos
        if acc_c is not None and ind_c is not None and meta_c is not None:
            acc_val  = row_vals[acc_c] if acc_c < ncols else ""
            ind_val  = row_vals[ind_c] if ind_c < ncols else ""
            meta_val = row_vals[meta_c] if meta_c < ncols else ""
            hay_algo = any([acc_val, ind_val, meta_val])

            # si la fila está vacía y no es encabezado ni títulos, terminar bloque de tabla
            if not hay_algo:
                i += 1
                continue

            # Guardar registro si tenemos una línea y/o problemática activa
            if cur_problem or cur_linea:
                registros.append({
                    "problematica": cur_problem,
                    "linea_accion": cur_linea,
                    "accion_estrategica": acc_val,
                    "indicador": ind_val,
                    "meta": meta_val,
                })
            i += 1
            continue

        i += 1

    # limpiar basura (filas totalmente vacías)
    df_out = pd.DataFrame(registros)
    if not df_out.empty:
        df_out = df_out.replace({"None": "", "nan": ""})
        df_out = df_out[~(df_out[["accion_estrategica","indicador","meta"]].fillna("").eq("").all(axis=1))]
    return df_out

def filtrar_municipal(df: pd.DataFrame, palabra_clave: str) -> pd.DataFrame:
    if not palabra_clave.strip():
        return df
    kw = palabra_clave.lower()
    mask = df.apply(lambda r: kw in (" ".join([str(x) for x in r.values])).lower(), axis=1)
    return df[mask]

# ------------------ PDF ------------------

def add_cover_from_path(c: canvas.Canvas, image_path: Optional[str]):
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
            # sin imagen no rompemos
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

def crear_pdf(regs: pd.DataFrame, image_path: Optional[str]) -> bytes:
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)

    # Portada
    add_cover_from_path(c, image_path)

    # Páginas de contenido
    total_pages_est = 1 + max(1, len(regs))  # estimación simple
    page = 1
    draw_header(c, page, total_pages_est)
    y = A4[1] - 3.6*cm
    x = 1.4*cm
    w = A4[0] - 2.8*cm

    idx = 1
    for _, r in regs.iterrows():
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

with st.sidebar:
    ruta_img = st.text_input("Ruta imagen portada (en tu repo)", value="assets/encabezado.png")
    palabra_muni = st.text_input("Filtro municipal (vacío = sin filtrar)", value="municip")

excel_file = st.file_uploader("Subí tu Excel de la matriz", type=["xlsx", "xls"])

if excel_file is None:
    st.info("Cargá el Excel para comenzar. La imagen se toma de la ruta local que indiques en la barra lateral.")
    st.stop()

# Leer (primera hoja)
try:
    df_raw = pd.read_excel(excel_file, engine="openpyxl", header=None)  # sin suponer encabezados
except Exception as e:
    st.error(f"No se pudo leer el Excel: {e}")
    st.stop()

regs = parse_matriz_dynamic(df_raw)

if regs.empty:
    st.warning("No se detectaron bloques. Verifica que existan celdas con 'Cadena de Resultados' y 'Línea de acción #', y que la fila de encabezados contenga 'Accion/Indicador/Meta'.")
else:
    # Filtro municipal
    regs_filtrado = filtrar_municipal(regs, palabra_muni) if palabra_muni else regs
    st.subheader("Vista previa (líneas incluidas)")
    st.dataframe(regs_filtrado, use_container_width=True)

    if st.button("Generar PDF editable"):
        pdf_bytes = crear_pdf(regs_filtrado, ruta_img if ruta_img else None)
        st.success("PDF generado.")
        st.download_button(
            label="⬇️ Descargar PDF",
            data=pdf_bytes,
            file_name="Informe_Seguimiento_GobiernoLocal.pdf",
            mime="application/pdf"
        )










