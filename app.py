# app.py — versión ajustada a tu matriz
import io, re, unicodedata
from typing import Dict, Optional, Tuple

import streamlit as st
import pandas as pd

# ReportLab (PDF)
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
FF_MULTILINE = 4096  # campo multilínea

st.set_page_config(page_title="PDF editable – Seguimiento GL", layout="wide")
st.title("Generar PDF editable – Gobierno Local (Sembremos Seguridad)")

st.markdown(
    "Carga el Excel de la **matriz**. Se detecta la fila de encabezado con "
    "**Acciones Estrategicas** y se toman las columnas (1=Acción, 2=Indicador, 4=Meta, 5=Líder, 6=Co-gestores). "
    "La **Problemática** y la **Línea de acción** se leen hacia arriba."
)

# ---------- Utils ----------
def _norm(s: str) -> str:
    if s is None: return ""
    s = str(s).strip().lower()
    return "".join(ch for ch in unicodedata.normalize("NFKD", s) if not unicodedata.combining(ch))

def encontrar_fila_header(df: pd.DataFrame) -> Tuple[int, Dict[str,int]]:
    """
    Busca la fila cuyo texto contiene 'Acciones Estrategicas' y mapea columnas:
    acción=1, indicador=2, meta=4, líder=5, cogestores=6 (0-index).
    Si no aparece exactamente, intenta sinónimos.
    """
    nrows, ncols = df.shape
    # búsqueda por palabras clave
    for i in range(nrows):
        row_low = [_norm(df.iat[i,j]) for j in range(ncols)]
        if any("acciones estrategica" in c or "acciones estrategicas" in c for c in row_low):
            # mapeo por nombres exactos si están; si no, usa los índices de tu archivo
            # (1,2,4,5,6). Ajusta si en tu plantilla cambian.
            mapping = {"accion": None, "indicador": None, "meta": None, "lider": None, "cogestores": None}
            # intenta localizar por texto
            for j, c in enumerate(row_low):
                if mapping["accion"] is None and ("acciones estrategica" in c or "acciones estrategicas" in c or "accion" in c):
                    mapping["accion"] = j
                if mapping["indicador"] is None and ("indicador" in c or "indicadores" in c):
                    mapping["indicador"] = j
                if mapping["meta"] is None and (re.search(r"\bmeta(s)?\b", c) or "efecto" in c):
                    mapping["meta"] = j
                if mapping["lider"] is None and ("lider estrategico" in c or "líder estratégico" in c or "lider" in c):
                    mapping["lider"] = j
                if mapping["cogestores"] is None and ("co-gestores" in c or "cogestores" in c or "co gestores" in c):
                    mapping["cogestores"] = j
            # fallback duro a tu estructura si algo quedó None
            mapping.setdefault("accion", 1);       mapping["accion"] = mapping["accion"] if mapping["accion"] is not None else 1
            mapping.setdefault("indicador", 2);    mapping["indicador"] = mapping["indicador"] if mapping["indicador"] is not None else 2
            mapping.setdefault("meta", 4);         mapping["meta"] = mapping["meta"] if mapping["meta"] is not None else 4
            mapping.setdefault("lider", 5);        mapping["lider"] = mapping["lider"] if mapping["lider"] is not None else 5
            mapping.setdefault("cogestores", 6);   mapping["cogestores"] = mapping["cogestores"] if mapping["cogestores"] is not None else 6
            return i, mapping
    return -1, {}

def contexto_superior(df: pd.DataFrame, hasta_fila: int) -> Dict[str,str]:
    """Busca hacia arriba 'Cadena de Resultados: ...' y 'Línea de acción #...'."""
    problem = ""; linea = ""
    for i in range(hasta_fila):
        for j in range(df.shape[1]):
            val = df.iat[i,j]
            if pd.isna(val): continue
            txt = str(val).strip(); low = _norm(txt)
            if low.startswith("cadena de resultados"):
                problem = txt.split(":",1)[-1].strip() if ":" in txt else txt
            if low.startswith("linea de accion") or low.startswith("línea de acción"):
                linea = txt
    return {"problematica": problem, "linea_accion": linea}

def es_muni(txt: str) -> bool:
    t = _norm(txt)
    return any(p in t for p in ["municip", "gobierno local", "alcald", "ayuntamiento"])

def parsear(df: pd.DataFrame) -> pd.DataFrame:
    """Parser específico para tu archivo."""
    S = df.astype(str).where(~df.isna(), "")
    hdr_row, cols = encontrar_fila_header(S)
    if hdr_row < 0:
        return pd.DataFrame(columns=["problematica","linea_accion","accion_estrategica","indicador","meta","lider","cogestores"])
    ctx = contexto_superior(S, hdr_row)
    cur_prob, cur_lin = ctx["problematica"], ctx["linea_accion"]
    regs = []
    nrows, ncols = S.shape

    for i in range(hdr_row+1, nrows):
        row = [S.iat[i,j] for j in range(ncols)]
        # si más abajo vuelve a aparecer una nueva línea o cadena, actualizar contexto
        for j in range(ncols):
            low = _norm(row[j])
            if low.startswith("linea de accion") or low.startswith("línea de acción"):
                cur_lin = row[j].strip()
            if low.startswith("cadena de resultados"):
                tx = row[j].strip(); cur_prob = tx.split(":",1)[-1].strip() if ":" in tx else tx
        def _get(idx): return row[idx].strip() if idx is not None and idx < ncols else ""
        acc  = _get(cols["accion"])
        ind  = _get(cols["indicador"])
        meta = _get(cols["meta"])
        lid  = _get(cols["lider"])
        cog  = _get(cols["cogestores"])
        if not any([acc, ind, meta]):  # fila vacía de datos
            continue
        regs.append({
            "problematica": cur_prob, "linea_accion": cur_lin,
            "accion_estrategica": acc, "indicador": ind, "meta": meta,
            "lider": lid, "cogestores": cog
        })
    return pd.DataFrame(regs).reset_index(drop=True)

# ---------- PDF helpers ----------
def add_cover(c: canvas.Canvas, image_path: Optional[str]):
    y_top = A4[1] - 2.5*cm
    if image_path:
        try:
            img = ImageReader(image_path)
            iw, ih = img.getSize(); max_w, max_h = A4[0]-4*cm, 7*cm
            scale = min(max_w/iw, max_h/ih); w, h = iw*scale, ih*scale
            x = (A4[0]-w)/2; y = y_top - h
            c.drawImage(img, x, y, width=w, height=h, preserveAspectRatio=True, mask='auto')
            y_top = y - 0.8*cm
        except Exception:
            pass
    c.setFillColor(AZUL_OSCURO); c.setFont("Helvetica-Bold", 28)
    c.drawCentredString(A4[0]/2, y_top-1.6*cm, "INFORME DE SEGUIMIENTO")
    c.setFont("Helvetica-Bold", 22); c.drawCentredString(A4[0]/2, y_top-3.1*cm, "Gobierno Local")
    c.setFont("Helvetica", 11); c.setFillColor(NEGRO); c.drawCentredString(A4[0]/2, y_top-4.6*cm, "Sembremos Seguridad")
    c.showPage()

def draw_header(c, page, total):
    c.setFillColor(AZUL_CLARO); c.rect(1*cm, A4[1]-2.6*cm, A4[0]-2*cm, 1.6*cm, fill=1, stroke=0)
    c.setFillColor(AZUL_OSCURO); c.setFont("Helvetica-Bold", 14)
    c.drawString(1.2*cm, A4[1]-1.8*cm, "Informe de Seguimiento – Gobierno Local | Sembremos Seguridad")
    c.setFont("Helvetica", 9); c.drawRightString(A4[0]-1.2*cm, A4[1]-1.3*cm, f"Página {page} de {total}")

def draw_footer(c):
    c.setFillColor(colors.grey); c.setLineWidth(0.5); c.line(1*cm, 1.8*cm, A4[0]-1*cm, 1.8*cm)
    c.setFillColor(NEGRO); c.setFont("Helvetica", 8); c.drawString(1.2*cm, 1.3*cm, "Evidencias deben agregarse en la carpeta compartida designada.")

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
    if y-need<2.8*cm:
        draw_footer(c); c.showPage(); draw_header(c, page+1, total); return A4[1]-3.6*cm, page+1
    return y, page

def crear_pdf(rows: pd.DataFrame, image_path: Optional[str]) -> bytes:
    buf = io.BytesIO(); c = canvas.Canvas(buf, pagesize=A4)
    add_cover(c, image_path)
    total_est = 1 + max(1, len(rows)); page=1; draw_header(c, page, total_est)
    y=A4[1]-3.6*cm; x=1.4*cm; w=A4[0]-2.8*cm; idx=1
    for _, r in rows.iterrows():
        y, page = ensure_space(c, y, 3.0*cm, page, total_est); y = section(c, x, y, w, "Cadena de Resultados / Problemática", r.get("problematica",""))
        y, page = ensure_space(c, y, 2.6*cm, page, total_est); y = section(c, x, y, w, "Línea de acción", r.get("linea_accion",""))
        y, page = ensure_space(c, y, 2.6*cm, page, total_est); y = section(c, x, y, w, "Acción Estratégica", r.get("accion_estrategica",""))
        y, page = ensure_space(c, y, 2.6*cm, page, total_est); y = section(c, x, y, w, "Indicador", r.get("indicador",""))
        y, page = ensure_space(c, y, 2.6*cm, page, total_est); y = section(c, x, y, w, "Meta", r.get("meta",""))
        alto=4.5*cm; y, page = ensure_space(c, y, alto+1.6*cm, page, total_est)
        c.setFillColor(AZUL_CLARO); c.rect(x, y-0.9*cm, w, 0.9*cm, fill=1, stroke=0)
        c.setFillColor(AZUL_OSCURO); c.setFont("Helvetica-Bold", 11); c.drawString(x+0.2*cm, y-0.6*cm, "Resultado (rellenable por Gobierno Local)")
        c.setStrokeColor(BORDE); c.rect(x, y-(alto+1.2*cm), w, alto, fill=0, stroke=1)
        c.acroForm.textfield(name=f"resultado_{idx}", tooltip=f"Resultado línea {idx}",
            x=x+0.15*cm, y=y-(alto+1.2*cm)+0.15*cm, width=w-0.3*cm, height=alto-0.3*cm,
            borderStyle="inset", borderWidth=1, forceBorder=True, fontName="Helvetica", fontSize=10, fieldFlags=FF_MULTILINE)
        y -= (alto+1.6*cm); idx+=1
    draw_footer(c); c.save(); buf.seek(0); return buf.read()

# ---------- UI ----------
with st.sidebar:
    ruta_img = st.text_input("Ruta imagen portada (en tu repo)", value="assets/encabezado.png")
    filtro = st.selectbox("Filtrado", ["Líder contiene municipalidad", "Líder o Co-gestores", "Sin filtro"])

excel = st.file_uploader("Subí tu Excel de la matriz", type=["xlsx","xls"])
if not excel:
    st.info("Cargá el Excel para comenzar."); st.stop()

# Leer tal cual (sin headers)
df = pd.read_excel(excel, engine="openpyxl", header=None)
regs = parsear(df)

# Aplicar filtro
if filtro == "Líder contiene municipalidad":
    regs_f = regs[regs["lider"].apply(lambda x: es_muni(x))]
elif filtro == "Líder o Co-gestores":
    regs_f = regs[regs.apply(lambda r: es_muni(r["lider"]) or es_muni(r["cogestores"]), axis=1)]
else:
    regs_f = regs

st.caption(f"Filas detectadas: {len(regs)} | Después del filtro: {len(regs_f)}")
st.subheader("Vista previa")
cols_show = ["problematica","linea_accion","accion_estrategica","indicador","meta","lider","cogestores"]
st.dataframe(regs_f[cols_show] if not regs_f.empty else regs[cols_show], use_container_width=True)

if st.button("Generar PDF editable"):
    data = regs_f if not regs_f.empty else regs
    pdf_bytes = crear_pdf(data, ruta_img or None)
    st.success("PDF generado.")
    st.download_button("⬇️ Descargar PDF", data=pdf_bytes, file_name="Informe_Seguimiento_GobiernoLocal.pdf", mime="application/pdf")






