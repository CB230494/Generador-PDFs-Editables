# app.py — PDF por contenido, con división por Indicador y filtro fijo "solo Municipalidad"
import io, re, unicodedata
from typing import Dict, List, Optional, Tuple
from pathlib import Path

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
- **Divide** una misma Acción en varias **fichas** (una por **Indicador**) alineando Meta/Líder por índice.  
- El **filtro es fijo**: *solo se incluyen fichas cuyo **Líder** sea municipal*.  
- En el PDF: **cada Problemática inicia una página nueva**; dentro se imprimen sus **Líneas** y sus **fichas**.
"""
)

# ========= Utilidades =========
def _norm(s: str) -> str:
    """Normaliza texto: minúsculas sin acentos/espacios extremos."""
    if s is None:
        return ""
    s = str(s).strip().lower()
    return "".join(ch for ch in unicodedata.normalize("NFKD", s) if not unicodedata.combining(ch))

# Sinónimos que aparecen en las matrices
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
    Ubica columnas por texto. Devuelve indices:
    accion, indicador, meta, lider, cogestores, consideraciones (algunas pueden no existir).
    Para considerar encabezado deben existir al menos acción/indicador/meta.
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

# --- Helpers para dividir campos múltiples y expandir filas ---
ITEM_SEP_REGEX = re.compile(
    r"(?:\r?\n|;|\s*/\s*|\s*\|\s*|•|\u2022)",  # saltos, ; , / , | , bullets
    flags=re.UNICODE
)

def _split_items(text: str) -> List[str]:
    """
    Divide 'text' en partes si parece lista múltiple.
    Soporta numeraciones '1) …', '2.' etc. y separadores comunes.
    """
    t = (text or "").strip()
    if not t:
        return []
    # Numeraciones tipo "1) foo" por líneas
    lines = [ln.strip() for ln in re.split(r"\r?\n", t) if ln.strip()]
    if any(re.match(r"^\d+[\)\.\-]\s+", ln) for ln in lines):
        parts, buf = [], ""
        for ln in lines:
            if re.match(r"^\d+[\)\.\-]\s+", ln):
                if buf.strip(): parts.append(buf.strip())
                buf = re.sub(r"^\d+[\)\.\-]\s+", "", ln)
            else:
                buf += (" " + ln)
        if buf.strip(): parts.append(buf.strip())
        return [p for p in parts if p]
    # Separadores comunes
    parts = [p.strip() for p in ITEM_SEP_REGEX.split(t) if p and p.strip()]
    return parts if len(parts) > 1 else [t]

def _align_lists(n: int, *lists: List[str]) -> List[List[str]]:
    """
    Alinea listas para tener longitud n. Si una lista es vacía, usa ''.
    Si es más corta, repite el último valor no vacío.
    """
    out = []
    for L in lists:
        L = list(L)
        if not L:
            out.append([""] * n)
            continue
        if len(L) < n:
            last = L[-1] if L else ""
            L = L + [last] * (n - len(L))
        out.append(L[:n])
    return out

def expand_action_row(base: Dict[str, str]) -> List[Dict[str, str]]:
    """
    Expande una fila con múltiples indicadores/metas/líderes en varias filas.
    Mantiene problematica, linea_accion y accion_estrategica.
    """
    ind_parts  = _split_items(base.get("indicador", ""))
    meta_parts = _split_items(base.get("meta", ""))
    lid_parts  = _split_items(base.get("lider", ""))
    cog_parts  = _split_items(base.get("cogestores", ""))

    n = max(1, len(ind_parts))
    meta_parts, lid_parts, cog_parts = _align_lists(n, meta_parts, lid_parts, cog_parts)

    rows = []
    for i in range(n):
        rows.append({
            "problematica":       base.get("problematica", ""),
            "linea_accion":       base.get("linea_accion", ""),
            "accion_estrategica": base.get("accion_estrategica", ""),
            "indicador":          ind_parts[i] if i < len(ind_parts) else "",
            "meta":               meta_parts[i],
            "lider":              lid_parts[i],
            "cogestores":         cog_parts[i],
            "hoja":               base.get("hoja", "")
        })
    return rows

# ========= Parser por hoja =========
def parse_sheet(df_raw: pd.DataFrame, sheet_name: str) -> pd.DataFrame:
    """
    Recorre toda la hoja:
      - Guarda última 'Cadena de Resultados' y 'Línea de acción #'
      - Detecta encabezado (sinónimos)
      - Extrae filas de datos y las EXPANDE por Indicador/Meta/Líder
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

        # Actualizar contexto si aparece Cadena/Línea
        for j in range(ncols):
            cell = row_vals[j]
            if RE_CADENA.match(cell):
                current_problem = cell.split(":",1)[-1].strip() if ":" in cell else cell.strip()
            if RE_LINEA.match(cell):
                current_linea = cell.strip()

        # Detectar encabezado
        hdr = find_header_in_row(row_vals)
        if hdr:
            # Filas de datos hacia abajo
            i += 1
            while i < nrows:
                row_vals = [S.iat[i, j].strip() for j in range(ncols)]
                # Contexto puede actualizarse más abajo
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

                if not any([acc, ind, meta]):  # fin bloque
                    break

                # -------- EXPANSIÓN por Indicador/Meta/Líder --------
                row_base = {
                    "problematica": current_problem,
                    "linea_accion": current_linea,
                    "accion_estrategica": acc,
                    "indicador": ind,
                    "meta": meta,
                    "lider": lid,
                    "cogestores": cog,
                    "hoja": sheet_name,
                }
                rows.extend(expand_action_row(row_base))
                # ---------------------------------------------------

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

# ========= Detección automática de imagen de portada =========
def autodetect_cover_image() -> Optional[str]:
    """
    Busca en el repo una imagen candidata a portada.
    Prioriza nombres que contengan: encabezado, portada, header, imagen23.
    Extensiones soportadas: png, jpg, jpeg, webp.
    """
    root = Path.cwd()
    patterns = ["*encabezado.*", "*portada.*", "*header.*", "*imagen23.*"]
    exts = {".png", ".jpg", ".jpeg", ".webp"}

    candidates: List[Path] = []
    for base in [root, root / "assets"]:
        if base.exists():
            for pat in patterns:
                candidates += list(base.glob(pat))
    if not candidates:
        for pat in patterns:
            candidates += list(root.rglob(pat))

    candidates = [p for p in candidates if p.suffix.lower() in exts]
    return str(candidates[0]) if candidates else None

# ========= Helpers PDF (agrupado por PROBLEMÁTICA, cada una inicia página) =========
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
    """Tarjeta por Acción con Resultado editable."""
    c.setStrokeColor(BORDE); c.setFillColor(colors.white)
    c.rect(x, y-5.8*cm, w, 5.8*cm, fill=0, stroke=1)

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

def build_pdf_grouped_by_problem(rows: pd.DataFrame, image_path: Optional[str]) -> bytes:
    """
    AGRUPA por PROBLEMÁTICA.
    - Cada problemática inicia en una página nueva (encabezado al tope).
    - Dentro, agrupa por Línea de acción: se imprime la línea una sola vez y sus fichas.
    """
    buf = io.BytesIO(); c = canvas.Canvas(buf, pagesize=A4)
    portada(c, image_path)

    # Agrupar primero por problemática (en orden de aparición)
    if rows.empty:
        groups = [("", rows)]
    else:
        probs_in_order = list(dict.fromkeys(rows["problematica"].fillna("").tolist()))
        groups = [(p, rows[rows["problematica"] == p]) for p in probs_in_order]

    total = 1 + max(1, len(groups))
    page = 0

    for prob, df_prob in groups:
        # NUEVA PÁGINA para esta problemática
        page += 1
        c.showPage() if page > 1 else None
        header(c, page, total)
        x = 1.4*cm; w = A4[0]-2.8*cm; y = A4[1]-3.6*cm

        # Bloque de problemática (arriba)
        y = section_bar(c, x, y, w, "Cadena de Resultados / Problemática")
        y = wrap_text(c, prob or "", x+0.2*cm, y, w-0.4*cm)
        y -= 0.2*cm

        if df_prob.empty:
            footer(c); continue

        lineas_in_order = list(dict.fromkeys(df_prob["linea_accion"].fillna("").tolist()))
        for lin in lineas_in_order:
            gdf = df_prob[df_prob["linea_accion"] == lin].reset_index(drop=True)

            y, page = ensure_space(c, y, 3.0*cm, page, total)
            y = section_bar(c, x, y, w, "Línea de acción")
            y = wrap_text(c, lin or "", x+0.2*cm, y, w-0.4*cm)
            y -= 0.3*cm

            # Fichas de acciones para esta línea
            for idx, fila in gdf.iterrows():
                y, page = ensure_space(c, y, 7.0*cm, page, total)
                y = ficha_accion(c, x, y, w, idx+1, fila)

        footer(c)

    c.save(); buf.seek(0); return buf.read()

# ========= UI =========
with st.sidebar:
    cover_path = autodetect_cover_image()
    if cover_path:
        st.success(f"Imagen detectada: `{Path(cover_path).as_posix()}`")
    else:
        st.warning("No se detectó imagen de portada. Colocá 'imagen23.*', 'encabezado.*', 'portada.*' o 'header.*' (png/jpg/jpeg/webp) en el repo.")

excel_file = st.file_uploader("Subí tu Excel (multipestaña o una sola)", type=["xlsx","xls"])
if not excel_file:
    st.info("Cargá el Excel para comenzar. La portada usa la imagen detectada automáticamente.")
    st.stop()

# Procesar hojas
xls_names = pd.ExcelFile(excel_file, engine="openpyxl").sheet_names
modo_hojas = st.radio("Hojas a procesar", ["Todas", "Elegir una"], horizontal=True)
if modo_hojas == "Elegir una":
    hoja_sel = st.selectbox("Hoja a procesar", xls_names)
    df_hoja = pd.read_excel(excel_file, sheet_name=hoja_sel, header=None, engine="openpyxl")
    regs = parse_sheet(df_hoja, hoja_sel)
else:
    regs = pd.concat([parse_sheet(pd.read_excel(excel_file, sheet_name=s, header=None, engine="openpyxl"), s)
                      for s in xls_names], ignore_index=True)

st.caption(f"Filas detectadas (todas las hojas seleccionadas): **{len(regs)}**")

# 🔒 Filtro FIJO: solo fichas cuyo LÍDER sea municipal
regs_f = regs[regs["lider"].apply(es_muni)].reset_index(drop=True)

st.caption(f"Filas después del filtro (solo Municipalidad): **{len(regs_f)}**")
st.subheader("Vista previa")
cols = ["hoja","problematica","linea_accion","accion_estrategica","indicador","meta","lider","cogestores"]
st.dataframe(regs_f[cols] if not regs_f.empty else regs[cols], use_container_width=True)

if st.button("Generar PDF editable"):
    data = regs_f if not regs_f.empty else regs.iloc[0:0]  # si no hay municipales, genera PDF vacío (sin errores)
    pdf_bytes = build_pdf_grouped_by_problem(data, cover_path)
    st.success("PDF generado.")
    st.download_button("⬇️ Descargar PDF", data=pdf_bytes, file_name="Informe_Seguimiento_GobiernoLocal.pdf", mime="application/pdf")



