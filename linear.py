# matrix_app.py
# Matrix Transformations — Single-file Streamlit app
# Requirements: streamlit, numpy, matplotlib, pandas
# Save as matrix_app.py and run: streamlit run matrix_app.py

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from io import BytesIO

st.set_page_config(page_title="Matrix Transformations", layout="wide")

# ---------------------------
# Language strings (EN / ID)
# ---------------------------
LANGS = {
    "en": {
        "title": "Matrix Transformations — Interactive Webapp",
        "desc": "Apply and visualize 2D affine and linear transforms.",
        "shape": "Choose base shape (or upload CSV)",
        "shapes": ["Unit square", "Triangle", "Custom (upload CSV)"],
        "upload_help": "CSV with two columns: x,y (with or without header).",
        "transform": "Transformation type",
        "translations": "Translation",
        "scaling": "Scaling",
        "rotation": "Rotation",
        "shearing": "Shearing",
        "reflection": "Reflection",
        "custom": "Custom matrix",
        "apply": "Apply transformation",
        "matrix": "Transformation matrix (3x3 homogeneous)",
        "original": "Original",
        "transformed": "Transformed",
        "download_points": "Download transformed points (CSV)",
        "download_code": "Download this Python file",
        "angle": "Angle (degrees)",
        "axis": "Axis",
        "axis_options": ["x", "y", "origin", "y=x"],
        "sx": "sx (scale x)",
        "sy": "sy (scale y)",
        "tx": "tx (translate x)",
        "ty": "ty (translate y)",
        "shx": "shear x (shx)",
        "shy": "shear y (shy)",
        "error_csv": "CSV must have at least two columns (x,y).",
        "no_points": "No points available to transform.",
    },
    "id": {
        "title": "Transformasi Matriks — Aplikasi Interaktif",
        "desc": "Terapkan dan visualisasikan transformasi 2D (afine & linear).",
        "shape": "Pilih bentuk dasar (atau unggah CSV)",
        "shapes": ["Persegi satuan", "Segitiga", "Kustom (unggah CSV)"],
        "upload_help": "CSV dengan dua kolom: x,y (dengan atau tanpa header).",
        "transform": "Jenis transformasi",
        "translations": "Translasi",
        "scaling": "Skalasi",
        "rotation": "Rotasi",
        "shearing": "Shearing",
        "reflection": "Refleksi",
        "custom": "Matriks kustom",
        "apply": "Terapkan transformasi",
        "matrix": "Matriks transformasi (3x3 homogen)",
        "original": "Asli",
        "transformed": "Tertransformasi",
        "download_points": "Unduh titik tertransformasi (CSV)",
        "download_code": "Unduh file Python ini",
        "angle": "Sudut (derajat)",
        "axis": "Sumbu",
        "axis_options": ["x", "y", "origin", "y=x"],
        "sx": "sx (skala x)",
        "sy": "sy (skala y)",
        "tx": "tx (translasi x)",
        "ty": "ty (translasi y)",
        "shx": "shear x (shx)",
        "shy": "shear y (shy)",
        "error_csv": "CSV harus memiliki minimal dua kolom (x,y).",
        "no_points": "Tidak ada titik untuk ditransformasikan.",
    }
}

# ---------------------------
# Utility: transformation matrices
# ---------------------------
def translation_matrix(tx, ty):
    return np.array([[1,0,tx],
                     [0,1,ty],
                     [0,0,1]], dtype=float)

def scaling_matrix(sx, sy):
    return np.array([[sx,0,0],
                     [0,sy,0],
                     [0,0,1]], dtype=float)

def rotation_matrix(deg):
    rad = np.deg2rad(deg)
    c = np.cos(rad); s = np.sin(rad)
    return np.array([[c,-s,0],
                     [s, c,0],
                     [0, 0,1]], dtype=float)

def shearing_matrix(shx, shy):
    return np.array([[1, shx, 0],
                     [shy, 1,  0],
                     [0,  0,  1]], dtype=float)

def reflection_matrix(axis):
    if axis == "x":
        return np.array([[1,0,0],[0,-1,0],[0,0,1]], dtype=float)
    if axis == "y":
        return np.array([[-1,0,0],[0,1,0],[0,0,1]], dtype=float)
    if axis == "origin":
        return np.array([[-1,0,0],[0,-1,0],[0,0,1]], dtype=float)
    if axis == "y=x":
        return np.array([[0,1,0],[1,0,0],[0,0,1]], dtype=float)
    raise ValueError("Unknown axis")

def apply_transform(points, M):
    # points: (N,2) numpy array
    if points is None or len(points) == 0:
        return np.zeros((0,2))
    hom = np.hstack([points, np.ones((points.shape[0],1))])
    res = hom @ M.T
    return res[:, :2]

# ---------------------------
# UI: language selection
# ---------------------------
col_lang = st.sidebar.selectbox("Language / Bahasa", ["English","Bahasa Indonesia"])
lang = "en" if col_lang == "English" else "id"
L = LANGS[lang]

st.title(L["title"])
st.write(L["desc"])

# ---------------------------
# Layout: controls (left) + visualization (right)
# ---------------------------
left, right = st.columns([1,2])

with left:
    st.subheader(L["shape"])
    shape_choice = st.selectbox("", L["shapes"])
    uploaded = None
    points = None
    if shape_choice == L["shapes"][-1]:  # custom upload
        uploaded = st.file_uploader(L["upload_help"], type=["csv","txt"])
        if uploaded:
            try:
                df = pd.read_csv(uploaded)
                if df.shape[1] < 2:
                    st.error(L["error_csv"])
                else:
                    # take first two columns
                    points = df.iloc[:, :2].to_numpy(dtype=float)
            except Exception as e:
                st.error(f"Failed to read CSV: {e}")
    else:
        # default shapes
        if shape_choice == L["shapes"][0]:  # unit square / persegi satuan
            points = np.array([[0.0,0.0],[1.0,0.0],[1.0,1.0],[0.0,1.0],[0.0,0.0]])
        else:  # triangle
            points = np.array([[0.0,0.0],[1.0,0.0],[0.5,0.8],[0.0,0.0]])

    st.markdown("---")
    st.subheader(L["transform"])
    ttype = st.selectbox("", [
        L["translations"],
        L["scaling"],
        L["rotation"],
        L["shearing"],
        L["reflection"],
        L["custom"]
    ])

    # Build matrix M based on selected transform
    M = np.eye(3)
    if ttype == L["translations"]:
        tx = st.number_input(L["tx"], value=0.0, format="%.6f")
        ty = st.number_input(L["ty"], value=0.0, format="%.6f")
        M = translation_matrix(tx, ty)
    elif ttype == L["scaling"]:
        sx = st.number_input(L["sx"], value=1.0, format="%.6f")
        sy = st.number_input(L["sy"], value=1.0, format="%.6f")
        M = scaling_matrix(sx, sy)
    elif ttype == L["rotation"]:
        deg = st.slider(L["angle"], -360.0, 360.0, 45.0)
        M = rotation_matrix(deg)
    elif ttype == L["shearing"]:
        shx = st.number_input(L["shx"], value=0.0, format="%.6f")
        shy = st.number_input(L["shy"], value=0.0, format="%.6f")
        M = shearing_matrix(shx, shy)
    elif ttype == L["reflection"]:
        axis = st.selectbox(L["axis"], L["axis_options"])
        M = reflection_matrix(axis)
    else:  # custom matrix
        st.write("Enter 9 numbers for a 3x3 homogeneous matrix (row-major).")
        cols_inp = st.columns(3)
        vals = []
        for r in range(3):
            with cols_inp[r]:
                a1 = st.number_input(f"a{r+1}1", value=float(M[r,0]))
                a2 = st.number_input(f"a{r+1}2", value=float(M[r,1]))
                a3 = st.number_input(f"a{r+1}3", value=float(M[r,2]))
                vals.extend([a1,a2,a3])
        M = np.array(vals).reshape(3,3)

    st.markdown(L["matrix"])
    st.code(np.array2string(M, precision=6, separator=', '))

    if st.button(L["apply"]):
        if points is None or len(points)==0:
            st.error(L["no_points"])
        else:
            transformed = apply_transform(points, M)
            st.session_state["orig_pts"] = points
            st.session_state["trans_pts"] = transformed

    # If not applied yet, compute preview transform
    if "orig_pts" not in st.session_state:
        st.session_state["orig_pts"] = points
        st.session_state["trans_pts"] = apply_transform(points, M)

    # Download transformed points
    df_out = pd.DataFrame(columns=["x_orig","y_orig","x_trans","y_trans"])
    if st.session_state.get("orig_pts") is not None:
        orig = st.session_state["orig_pts"]
        trans = st.session_state["trans_pts"]
        df_out = pd.DataFrame({
            "x_orig": np.round(orig[:,0],6),
            "y_orig": np.round(orig[:,1],6),
            "x_trans": np.round(trans[:,0],6),
            "y_trans": np.round(trans[:,1],6),
        })
        csv_bytes = df_out.to_csv(index=False).encode("utf-8")
        st.download_button(L["download_points"], data=csv_bytes, file_name="transformed_points.csv", mime="text/csv")

    # Download this Python file (attempt)
    try:
        # reading own file (works when run from file)
        with open(__file__, "r", encoding="utf-8") as f:
            code_text = f.read()
        st.download_button(L["download_code"], data=code_text, file_name="matrix_app.py", mime="text/x-python")
    except Exception:
        # fallback: provide minimal instructions
        st.info("If download button not available in your environment, copy the content of this app file into a new file named matrix_app.py.")

with right:
    st.subheader("Visualization")
    orig = st.session_state.get("orig_pts")
    trans = st.session_state.get("trans_pts")
    if orig is None:
        st.write(L["no_points"])
    else:
        fig, ax = plt.subplots(figsize=(6,6))
        ax.set_aspect('equal', adjustable='box')
        # axes
        ax.axhline(0, linewidth=0.5)
        ax.axvline(0, linewidth=0.5)
        # original shape
        ax.plot(orig[:,0], orig[:,1], marker='o', linestyle='-', label=L["original"])
        # transformed shape
        ax.plot(trans[:,0], trans[:,1], marker='o', linestyle='--', label=L["transformed"])
        ax.legend()
        ax.grid(True)
        st.pyplot(fig)

        st.subheader("Coordinates")
        st.dataframe(df_out)

# Footer / help
st.markdown("---")
st.write("Notes: This app supports Translation, Scaling, Rotation, Shearing, Reflection, and Custom 3x3 homogeneous matrices.")
st.write("To deploy: push this file to GitHub and use Share on Streamlit (share.streamlit.io).")
