# app.py — "Gobierno Local de <CANTÓN>" desde el nombre del archivo (toma el segmento antes de " - ")
import io, re, unicodedata
from typing import Dict, List, Optional
from pathlib import Path

import streamlit as st
import pandas as pd

# PDF
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.lib.utils import ImageReader

# ===== Estilo =====
AZUL_OSCURO = colors.HexColor("#1F4E79")
AZUL_CLARO  = colors.HexColor("#DCEBF7")
BORDE       = colors.HexColor("#9BBBD9")
NEGRO       = colors.black
FF_MULTILINE = 4096
MAXLEN_MUY_GRANDE = 100000  # ⬅️ para quitar límites prácticos de caracteres

st.set_page_config(page_title="PDF editable – Gobierno Local", layout="wide")
st.title("Generar PDF editable – Gobierno Local (Sembremos Seguridad)")

st.markdown("""
- Detecta por **contenido**: *Cadena de Resultados*, *Línea de acción #*, encabezados (Acciones, Indicador/Productos-Servicios, Meta/Efectos, Líder, Co-gestores).
- **Divide** una acción en **múltiples fichas** (una por **Indicador**) y **arrastra** Acción/Meta/Líder dentro del bloque si vienen en celdas combinadas.
- **Sólo** incluye fichas con **Líder municipal**.
- Numeración **simple**: **Acción #1, #2, #3, …** (sin 2.1).
- Portada autodetectada (`imagen23.*`, `encabezado.*`, `portada.*`, `header.*`) y subtítulo **“Gobierno Local de <CANTÓN>”** desde el nombre del archivo.
""")

# ========= Utils =========
def _norm(s: str) -> str:
    if s is None: return ""
    s = str(s).strip().lower()
    return "".join(ch for ch in unicodedata.normalize("NFKD", s) if not unicodedata.combining(ch))

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

# --- División por múltiples ítems en una sola celda ---
ITEM_SEP_REGEX = re.compile(r"(?:\r?\n|;|\s*/\s*|\s*\|\s*|•|\u2022)", flags=re.UNICODE)

def _split_items(text: str) -> List[str]:
    t = (text or "").strip()
    if not t: return []
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
    parts = [p.strip() for p in ITEM_SEP_REGEX.split(t) if p and p.strip()]
    return parts if len(parts) > 1 else [t]

def _align_lists(n: int, *lists: List[str]) -> List[List[str]]:
    out = []
    for L in lists:
        L = list(L)
        if not L: out.append([""] * n); continue
        if len(L) < n:
            last = L[-1] if L else ""
            L = L + [last] * (n - len(L))
        out.append(L[:n])
    return out

def expand_action_row(base: Dict[str, str]) -> List[Dict[str, str]]:
    """Si una celda trae varios indicadores, crea N filas (alineando meta/líder por índice)."""
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

# ========= Parser =========
def parse_sheet(df_raw: pd.DataFrame, sheet_name: str) -> pd.DataFrame:
    S = df_raw.astype(str).where(~df_raw.isna(), "")
    nrows, ncols = S.shape
    current_problem, current_linea = "", ""

    # forward-fill dentro del bloque
    last_action, last_meta, last_lider, last_cog = "", "", "", ""

    rows: List[Dict] = []
    i = 0
    while i < nrows:
        row_vals = [S.iat[i, j].strip() for j in range(ncols)]

        for j in range(ncols):
            cell = row_vals[j]
            if RE_CADENA.match(cell):
                current_problem = cell.split(":",1)[-1].strip() if ":" in cell else cell.strip()
            if RE_LINEA.match(cell):
                current_linea = cell.strip()

        hdr = find_header_in_row(row_vals)
        if hdr:
            i += 1
            # reiniciar arrastres al entrar a una nueva tabla
            last_action = last_meta = last_lider = last_cog = ""
            while i < nrows:
                row_vals = [S.iat[i, j].strip() for j in range(ncols)]

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

                # ⬇️ CAMBIO: filas sep./vacías dentro del bloque -> saltar, NO terminar
                if not any([acc, ind, meta, lid, cog]):
                    i += 1
                    continue

                # forward-fill por bloque
                if acc: last_action = acc
                if meta: last_meta = meta
                if lid:  last_lider = lid
                if cog:  last_cog   = cog

                base = {
                    "problematica": current_problem,
                    "linea_accion": current_linea,
                    "accion_estrategica": last_action,
                    "indicador": ind,
                    "meta": last_meta,
                    "lider": last_lider,
                    "cogestores": last_cog,
                    "hoja": sheet_name,
                }
                rows.extend(expand_action_row(base))
                i += 1
            continue
        i += 1

    return pd.DataFrame(rows)

def parse_workbook(xls_bytes) -> pd.DataFrame:
    xls = pd.ExcelFile(xls_bytes, engine="openpyxl")
    todo = []
    for sheet in xls.sheet_names:
        df = pd.read_excel(xls, sheet_name=sheet, header=None, engine="openpyxl")
        parsed = parse_sheet(df, sheet)
        if not parsed.empty: todo.append(parsed)
    return pd.concat(todo, ignore_index=True) if todo else pd.DataFrame(
        columns=["problematica","linea_accion","accion_estrategica","indicador","meta","lider","cogestores","hoja"]
    )

# ========= Portada auto =========
def autodetect_cover_image() -> Optional[str]:
    root = Path.cwd()
    patterns = ["*imagen23.*", "*encabezado.*", "*portada.*", "*header.*"]
    exts = {".png", ".jpg", ".jpeg", ".webp"}
    cands: List[Path] = []
    for base in [root, root/"assets"]:
        if base.exists():
            for p in patterns:
                cands += list(base.glob(p))
    if not cands:
        for p in patterns:
            cands += list(root.rglob(p))
    cands = [p for p in cands if p.suffix.lower() in exts]
    return str(cands[0]) if cands else None

# ========= "Gobierno Local de <CANTÓN>" desde el nombre del archivo =========
def guess_canton_from_filename(filename: str) -> str:
    """
    Regla prioritaria: si el nombre del archivo contiene '  ESS' (dos espacios seguidos, case-insensitive),
    se toma TODO lo que esté a la izquierda como el nombre del cantón.
    Ejemplo: 'Montes de Oca  ESS T1 2025.xlsx' -> 'Montes de Oca'
    """
    base = Path(filename).stem.strip()

    m = re.split(r"\s{2,}ESS\b", base, flags=re.IGNORECASE)
    if len(m) >= 2:
        left = m[0].strip().replace("_", " ")
        return " ".join(w.capitalize() if not re.match(r"^[A-ZÁÉÍÓÚÑ]{2,}$", w) else w for w in left.split())

    base2 = base.replace("_", " ")
    if " - " in base2:
        left = base2.split(" - ")[0].strip()
        return " ".join(s.capitalize() for s in left.split())

    seg = re.sub(r"\[.*?\]", "", base2, flags=re.U)
    seg = re.sub(r"(?i)\b(matriz|cadena|resultados|lineas|líneas|accion|acción|version|versi[oó]n|final|modificado|oficial|vista|protegida|d\d+)\b", "", seg)
    tokens = [t for t in seg.split() if re.match(r"[A-Za-zÁÉÍÓÚÑáéíóúñ]", t)]
    return tokens[0].capitalize() if tokens else ""

# ========= PDF helpers =========
def portada(c: canvas.Canvas, image_path: Optional[str], canton: str):
    y_top = A4[1] - 2.5*cm
    if image_path:
        try:
            img = ImageReader(image_path)
            iw, ih = img.getSize()
            scale = min((A4[0]-4*cm)/iw, (7*cm)/ih)
            w, h = iw*scale, ih*scale
            x = (A4[0]-w)/2; y = y_top - h
            c.drawImage(img, x, y, width=w, height=h, preserveAspectRatio=True, mask='auto')
            y_top = y - 0.8*cm
        except Exception:
            pass
    c.setFillColor(AZUL_OSCURO)
    c.setFont("Helvetica-Bold", 28); c.drawCentredString(A4[0]/2, y_top-1.6*cm, "INFORME DE SEGUIMIENTO")
    c.setFont("Helvetica-Bold", 22)
    titulo = f"Gobierno Local de {canton}" if canton else "Gobierno Local"
    c.drawCentredString(A4[0]/2, y_top-3.1*cm, titulo)
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

def ficha_accion(c, x, y, w, idx, fila, field_name: str) -> float:
    """Tarjeta por Acción con Resultado editable (campo independiente)."""
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
        name=field_name,
        tooltip=f"Resultado acción {idx}",
        x=x+0.25*cm, y=y_text-(alto+1.0*cm)+0.1*cm,
        width=w-0.5*cm, height=alto-0.2*cm,
        borderStyle="inset", borderWidth=1, forceBorder=True,
        fontName="Helvetica", fontSize=10,
        fieldFlags=FF_MULTILINE,
        maxlen=MAXLEN_MUY_GRANDE
    )
    return y_text-(alto+1.4*cm)

# Página final con periodo
def trimestre_page(c: canvas.Canvas, page: int, total: int):
    c.showPage()
    header(c, page, total)
    x = 1.4*cm; w = A4[0]-2.8*cm; y = A4[1]-3.6*cm

    y = section_bar(c, x, y, w, "Periodo del informe")
    y = wrap_text(c, "Indique el/los trimestre(s) a los que corresponde este informe (por ejemplo: T1 2025, T1–T2 2025).",
                  x+0.2*cm, y, w-0.4*cm)

    alto = 3.0*cm
    c.setStrokeColor(BORDE)
    c.rect(x+0.2*cm, y-(alto+0.6*cm), w-0.4*cm, alto, fill=0, stroke=1)
    c.acroForm.textfield(
        name="periodo_trimestres",
        tooltip="Periodo del informe (trimestres)",
        x=x+0.25*cm, y=y-(alto+0.6*cm)+0.1*cm,
        width=w-0.5*cm, height=alto-0.2*cm,
        borderStyle="inset", borderWidth=1, forceBorder=True,
        fontName="Helvetica", fontSize=11,
        fieldFlags=FF_MULTILINE,
        maxlen=MAXLEN_MUY_GRANDE
    )
    footer(c)

def build_pdf_grouped_by_problem(rows: pd.DataFrame, image_path: Optional[str], canton: str) -> bytes:
    buf = io.BytesIO(); c = canvas.Canvas(buf, pagesize=A4)
    portada(c, image_path, canton)

    if rows.empty:
        groups = [("", rows)]
    else:
        probs_order = list(dict.fromkeys(rows["problematica"].fillna("").tolist()))
        groups = [(p, rows[rows["problematica"] == p]) for p in probs_order]

    total = 1 + max(1, len(groups)) + 1
    page = 0
    unique_field_counter = 0

    for prob, df_prob in groups:
        page += 1
        c.showPage() if page > 1 else None
        header(c, page, total)
        x = 1.4*cm; w = A4[0]-2.8*cm; y = A4[1]-3.6*cm

        y = section_bar(c, x, y, w, "Cadena de Resultados / Problemática")
        y = wrap_text(c, prob or "", x+0.2*cm, y, w-0.4*cm)
        y -= 0.2*cm

        if df_prob.empty:
            footer(c); continue

        lineas_order = list(dict.fromkeys(df_prob["linea_accion"].fillna("").tolist()))
        for lin in lineas_order:
            gdf = df_prob[df_prob["linea_accion"] == lin].reset_index(drop=True)

            y, page = ensure_space(c, y, 3.0*cm, page, total)
            y = section_bar(c, x, y, w, "Línea de acción")
            y = wrap_text(c, lin or "", x+0.2*cm, y, w-0.4*cm)
            y -= 0.3*cm

            num_simple = 1
            for _, fila in gdf.iterrows():
                y, page = ensure_space(c, y, 7.0*cm, page, total)
                unique_field_counter += 1
                field_name = f"resultado_{unique_field_counter}"
                y = ficha_accion(c, x, y, w, str(num_simple), fila, field_name)
                num_simple += 1

        footer(c)

    page += 1
    trimestre_page(c, page, total)

    c.save(); buf.seek(0); return buf.read()

# ========= UI =========
with st.sidebar:
    cover_path = autodetect_cover_image()
    if cover_path:
        st.success(f"Portada: `{Path(cover_path).as_posix()}`")
    else:
        st.warning("Colocá `imagen23.*`, `encabezado.*`, `portada.*` o `header.*` (png/jpg/jpeg/webp) en el repo.")

excel_file = st.file_uploader("Subí tu Excel (multi-hoja o una sola)", type=["xlsx","xls"])
if not excel_file:
    st.info("Cargá el Excel para comenzar.")
    st.stop()

# Nombre del cantón desde el nombre del archivo
canton_name = guess_canton_from_filename(excel_file.name)

xls = pd.ExcelFile(excel_file, engine="openpyxl")
modo = st.radio("Hojas a procesar", ["Todas", "Elegir una"], horizontal=True)
if modo == "Elegir una":
    hoja = st.selectbox("Hoja", xls.sheet_names)
    regs = parse_sheet(pd.read_excel(excel_file, sheet_name=hoja, header=None, engine="openpyxl"), hoja)
else:
    regs = pd.concat([parse_sheet(pd.read_excel(excel_file, sheet_name=s, header=None, engine="openpyxl"), s)
                     for s in xls.sheet_names], ignore_index=True)

st.caption(f"Filas detectadas: **{len(regs)}**")

# ===== Clave canónica para agrupar acciones =====
def _accion_key(s: str) -> str:
    # normaliza NBSP y tildes, quita numeración/puntuación inicial y colapsa espacios
    s = unicodedata.normalize("NFKD", str(s or "")).replace("\u00A0", " ")
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = re.sub(r"^\s*\d+[\)\.\-]*\s*", "", s, flags=re.UNICODE)
    s = re.sub(r"\s+", " ", s, flags=re.UNICODE)
    return s.strip().lower()

regs = regs.copy()
regs["accion_key"] = regs["accion_estrategica"].apply(_accion_key)

# Filtro por ACCIÓN: si alguna subfila tiene líder municipal, TODA la acción pasa
mask_por_accion = regs.groupby("accion_key")["lider"].transform(lambda col: any(es_muni(v) for v in col))
regs_muni = regs[mask_por_accion].copy().reset_index(drop=True)

# ==== Normalización de roles (swap inteligente) ====
def _split_clean_list(text: str) -> List[str]:
    parts = [p.strip() for p in ITEM_SEP_REGEX.split(str(text or "")) if p and p.strip()]
    seen = set(); out = []
    for p in parts:
        k = _norm(p)
        if k not in seen:
            seen.add(k); out.append(p)
    return out

def _join_list(parts: List[str]) -> str:
    return "; ".join([p for p in parts if p])

def _has_muni_in_list(parts: List[str]) -> bool:
    return any(es_muni(p) for p in parts)

def _normalize_roles(row: pd.Series) -> pd.Series:
    lider = str(row.get("lider", "") or "")
    cog   = str(row.get("cogestores", "") or "")
    cog_list = _split_clean_list(cog)
    lider_is_muni = es_muni(lider)
    cog_has_muni  = _has_muni_in_list(cog_list)

    if lider_is_muni:
        # Mantener "Municipalidad" como líder y quitarlo de co-gestores
        cog_list = [p for p in cog_list if not es_muni(p)]
        row["lider"] = "Municipalidad"
        row["cogestores"] = _join_list(cog_list)
        return row

    if cog_has_muni:
        # Swap: Municipalidad pasa a líder, líder original baja a co-gestores (sin duplicados)
        cog_list = [p for p in cog_list if not es_muni(p)]
        if lider and _norm(lider) not in {_norm(p) for p in cog_list}:
            cog_list.append(lider)
        row["lider"] = "Municipalidad"
        row["cogestores"] = _join_list(cog_list)
        return row

    # No había Municipalidad → promovemos y conservamos el líder original en co-gestores
    if lider and _norm(lider) not in {_norm(p) for p in cog_list}:
        cog_list.append(lider)
    row["lider"] = "Municipalidad"
    row["cogestores"] = _join_list(cog_list)
    return row

regs_muni = regs_muni.apply(_normalize_roles, axis=1)

st.caption(f"Filas después del filtro (acciones con al menos un líder municipal): **{len(regs_muni)}**")

st.subheader("Vista previa")
cols = ["hoja","problematica","linea_accion","accion_estrategica","indicador","meta","lider","cogestores"]
st.dataframe(regs_muni[cols], use_container_width=True)

if st.button("Generar PDF editable"):
    pdf = build_pdf_grouped_by_problem(regs_muni, cover_path, canton_name)
    st.success("PDF generado.")
    st.download_button("⬇️ Descargar PDF",
                       data=pdf,
                       file_name=f"Informe_Seguimiento_GobiernoLocal_{canton_name or 'PDF'}.pdf",
                       mime="application/pdf")


