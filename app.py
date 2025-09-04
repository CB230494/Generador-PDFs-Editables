# app.py (versión robusta)
import io, re, unicodedata
from typing import List, Dict, Optional, Tuple

import streamlit as st
import pandas as pd

# PDF
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.lib.utils import ImageReader

# ===== Paleta =====
AZUL_OSCURO = colors.HexColor("#1F4E79")
AZUL_CLARO  = colors.HexColor("#DCEBF7")
BORDE       = colors.HexColor("#9BBBD9")
NEGRO       = colors.black
FF_MULTILINE = 4096

st.set_page_config(page_title="PDF editable – Seguimiento GL", layout="wide")
st.title("Generar PDF editable – Gobierno Local (Sembremos Seguridad)")

st.markdown("""Carga tu Excel. El sistema detecta Problemática, Línea, Acción, Indicador y Meta.
Podés elegir cómo filtrar por municipalidad (estricto por Líder, amplio por Líder o Co-gestores, o sin filtro).""")

# ---------- utils ----------
def _norm(s: str) -> str:
    if s is None: return ""
    s = str(s).strip().lower()
    s = "".join(ch for ch in unicodedata.normalize("NFKD", s) if not unicodedata.combining(ch))
    return s

SINONIMOS_ACCION   = ["acciones estrategicas", "acciones estrategica", "accion estrategica", "accion", "acciones"]
SINONIMOS_INDIC    = ["indicador", "indicadores", "productos/servicios", "producto/servicio", "producto", "servicio"]
SINONIMOS_META     = ["meta", "metas", "efecto", "efectos", "resultados esperados"]
SINONIMOS_LIDER    = ["lider estrategico", "líder estratégico", "lider", "responsable"]
SINONIMOS_COGE     = ["co-gestores", "cogestores", "co gestores", "co gestores:"]

def buscar_fila_encabezados(df: pd.DataFrame) -> Tuple[int, Dict[str,int]]:
    nrows, ncols = df.shape
    for i in range(nrows):
        row = ["" if pd.isna(df.iat[i,j]) else str(df.iat[i,j]) for j in range(ncols)]
        low = [_norm(x) for x in row]
        def _find(any_of):
            for j,c in enumerate(low):
                if any(k in c for k in any_of):
                    return j
            return None
        cA = _find(SINONIMOS_ACCION)
        cI = _find(SINONIMOS_INDIC)
        cM = _find(SINONIMOS_META)
        if cA is not None and cI is not None and cM is not None:
            cL = _find(SINONIMOS_LIDER)
            cC = _find(SINONIMOS_COGE)
            return i, {"accion": cA, "indicador": cI, "meta": cM, "lider": cL, "cogestores": cC}
    return -1, {}

def extraer_contexto_superior(df: pd.DataFrame, hasta_fila: int) -> Dict[str,str]:
    prob = ""; linea = ""
    for i in range(hasta_fila):
        for j in range(df.shape[1]):
            val = df.iat[i,j]; 
            if pd.isna(val): continue
            txt = str(val).strip(); low = _norm(txt)
            if low.startswith("cadena de resultados"):
                prob = txt.split(":",1)[-1].strip() if ":" in txt else txt
            if low.startswith("linea de accion") or low.startswith("línea de acción"):
                linea = txt
    return {"problematica": prob, "linea_accion": linea}

def parsear_matriz(df: pd.DataFrame) -> pd.DataFrame:
    S = df.astype(str).where(~df.isna(), "")
    fila_hdr, cols = buscar_fila_encabezados(S)
    if fila_hdr < 0: 
        return pd.DataFrame(columns=["problematica","linea_accion","accion_estrategica","indicador","meta","lider","cogestores"])
    ctx = extraer_contexto_superior(S, fila_hdr)
    cur_problem, cur_linea = ctx["problematica"], ctx["linea_accion"]
    regs = []
    nrows, ncols = S.shape

    for i in range(fila_hdr+1, nrows):
        row = [S.iat[i,j] for j in range(ncols)]
        # actualizar contexto si aparecen nuevos rótulos más abajo
        for j in range(ncols):
            low = _norm(row[j])
            if low.startswith("linea de accion") or low.startswith("línea de acción"):
                cur_linea = row[j].strip()
            if low.startswith("cadena de resultados"):
                txt = row[j].strip()
                cur_problem = txt.split(":",1)[-1].strip() if ":" in txt else txt

        def _get(idx): 
            return row[idx].strip() if idx is not None and idx < ncols else ""
        acc  = _get(cols.get("accion"))
        ind  = _get(cols.get("indicador"))
        meta = _get(cols.get("meta"))
        lid  = _get(cols.get("lider"))
        cog  = _get(cols.get("cogestores"))

        if not any([acc,ind,meta]): 
            continue

        regs.append({
            "problematica": cur_problem, "linea_accion": cur_linea,
            "accion_estrategica": acc, "indicador": ind, "meta": meta,
            "lider": lid, "cogestores": cog
        })
    return pd.DataFrame(regs).reset_index(drop=True)

def es_muni(txt: str) -> bool:
    t = _norm(txt)
    return any(p in t for p in ["municip", "gobierno local", "alcald", "ayuntamiento"])

# ---------- PDF ----------
def portada_con_imagen(c: canvas.Canvas, image_path: Optional[str]):
    y_top = A4[1] - 2.5*cm
    if image_path:
        try:
            img = ImageReader(image_path)
            iw, ih = img.getSize()
            max_w, max_h = A4[0]-4*cm, 7*cm
            scale = min(max_w/iw, max_h/ih)
            w, h = iw*scale, ih*scale
            x = (A4[0]-w)/2
            y = y_top - h
            c.drawImage(img, x, y, width=w, height=h, preserveAspectRatio=True, mask='auto')
            y_top = y - 0.8*cm
        except Exception:
            pass
    c.setFillColor(AZUL_OSCURO)
    c.setFont("Helvetica-Bold", 28); c.drawCentredString(A4[0]/2, y_top-1.6*cm, "INFORME DE SEGUIMIENTO")
    c.setFont("Helvetica-Bold", 22); c.drawCentredString(A4[0]/2, y_top-3.1*cm, "Gobierno Local")
    c.setFont("Helvetica", 11); c.setFillColor(NEGRO); c.drawCentredString(A4[0]/2, y_top-4.6*cm, "Sembremos Seguridad")
    c.showPage()

def draw_header(c: canvas.Canvas, page: int, total: int):
    c.setFillColor(AZUL_CLARO); c.rect(1*cm, A4[1]-2.6*cm, A4[0]-2*cm, 1.6*cm, fill=1, stroke=0)
    c.setFillColor(AZUL_OSCURO); c.setFont("Helvetica-Bold", 14)
    c.drawString(1.2*cm, A4[1]-1.8*cm, "Informe de Seguimiento – Gobierno Local | Sembremos Seguridad")
    c.setFont("Helvetica", 9); c.drawRightString(A4[0]-1.2*cm, A4[1]-1.3*cm, f"Página {page} de {total}")

def draw_footer(c: canvas.Canvas):
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

def crear_pdf(df_rows: pd.DataFrame, image_path: Optional[str]) -> bytes:
    buf = io.BytesIO(); c = canvas.Canvas(buf, pagesize=A4)
    portada_con_imagen(c, image_path)
    total_est = 1 + max(1, len(df_rows)); page=1; draw_header(c, page, total_est)
    y=A4[1]-3.6*cm; x=1.4*cm; w=A4[0]-2.8*cm; idx=1
    for _, r in df_rows.iterrows():
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
    ruta_img = st.text_input("Ruta imagen de portada (en tu repo)", value="assets/encabezado.png")
    modo_filtro = st.selectbox("Filtrar por municipalidad", ["Líder contiene municipalidad (estricto)","Líder o Co-gestores (amplio)","Sin filtro"])

excel = st.file_uploader("Subí tu Excel de la matriz", type=["xlsx","xls"])
if not excel:
    st.info("Cargá el Excel para comenzar."); st.stop()

df = pd.read_excel(excel, engine="openpyxl", header=None)
regs = parsear_matriz(df)

st.caption(f"Filas detectadas en la tabla: **{len(regs)}**")

# aplicar filtro elegido
if modo_filtro != "Sin filtro":
    if modo_filtro.startswith("Líder contiene"):
        regs_f = regs[regs["lider"].apply(es_muni)]
    else:
        regs_f = regs[(regs["lider"].apply(es_muni)) | (regs["cogestores"].apply(es_muni))]
else:
    regs_f = regs

st.caption(f"Filas después del filtro: **{len(regs_f)}**")
st.subheader("Vista previa")
st.dataframe(regs_f[["problematica","linea_accion","accion_estrategica","indicador","meta","lider","cogestores"]], use_container_width=True)

if len(regs_f)==0:
    st.warning("No hay filas tras el filtro actual. Probá 'Sin filtro' o 'Líder o Co-gestores (amplio)' para verificar los datos detectados.")

if st.button("Generar PDF editable"):
    pdf_bytes = crear_pdf(regs_f if len(regs_f)>0 else regs, ruta_img or None)
    st.success("PDF generado.")
    st.download_button("⬇️ Descargar PDF", data=pdf_bytes, file_name="Informe_Seguimiento_GobiernoLocal.pdf", mime="application/pdf")









