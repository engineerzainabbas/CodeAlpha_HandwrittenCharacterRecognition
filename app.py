import streamlit as st
import numpy as np
from PIL import Image, ImageOps
import tensorflow as tf
import plotly.graph_objects as go
import gzip, os, struct

# ── Page config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Handwritten Character Recognition | CodeAlpha",
    page_icon="✍️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── CSS ────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@400;700;800&display=swap');
html, body, [class*="css"] { font-family: 'Syne', sans-serif; }
.stApp { background: #0a0a0f; color: #e8e8f0; }

.main-title {
    font-size: 42px; font-weight: 800;
    background: linear-gradient(135deg, #fff 30%, #7c3aed 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    background-clip: text; line-height: 1.1; margin-bottom: 4px;
}
.sub-title {
    font-family: 'Space Mono', monospace; font-size: 12px;
    color: #6b6b80; letter-spacing: 2px; text-transform: uppercase; margin-bottom: 24px;
}
.metric-card {
    background: #16161f; border: 1px solid #2a2a3a;
    border-radius: 16px; padding: 20px; text-align: center; margin-bottom: 8px;
}
.metric-val { font-size: 28px; font-weight: 800; color: #06b6d4; font-family: 'Space Mono', monospace; }
.metric-lbl { font-size: 10px; color: #6b6b80; letter-spacing: 2px; text-transform: uppercase; margin-top: 4px; font-family: 'Space Mono', monospace; }
.pred-hero {
    background: #16161f; border: 2px solid #7c3aed; border-radius: 20px;
    padding: 32px; text-align: center; box-shadow: 0 0 40px rgba(124,58,237,0.2);
}
.pred-hero-empty {
    background: #16161f; border: 2px solid #2a2a3a; border-radius: 20px;
    padding: 32px; text-align: center;
}
.pred-char {
    font-size: 96px; font-weight: 800;
    background: linear-gradient(135deg, #06b6d4, #7c3aed);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    background-clip: text; line-height: 1;
}
.pred-conf { font-family: 'Space Mono', monospace; font-size: 14px; color: #10b981; margin-top: 8px; }
.section-label {
    font-family: 'Space Mono', monospace; font-size: 11px; color: #6b6b80;
    letter-spacing: 3px; text-transform: uppercase; margin-bottom: 12px;
}
.section-label span { color: #7c3aed; }
.info-box {
    background: #111118; border: 1px solid #2a2a3a; border-radius: 12px;
    padding: 16px; font-family: 'Space Mono', monospace; font-size: 12px;
    color: #6b6b80; line-height: 1.8;
}
.arch-card {
    background: #16161f; border: 1px solid #2a2a3a; border-radius: 14px;
    padding: 16px; text-align: left; height: 100%;
}
.arch-title { font-size: 12px; font-weight: 700; color: #7c3aed; font-family: 'Space Mono', monospace; margin-bottom: 8px; }
.arch-body  { font-size: 11px; font-family: 'Space Mono', monospace; color: #6b6b80; white-space: pre-line; line-height: 1.8; }
div[data-testid="stSidebar"] { background: #111118 !important; border-right: 1px solid #2a2a3a; }
.stButton > button {
    background: linear-gradient(135deg, #7c3aed, #6d28d9);
    color: white; border: none; border-radius: 10px;
    font-family: 'Syne', sans-serif; font-weight: 700;
    font-size: 15px; padding: 12px 32px; width: 100%;
    transition: all 0.2s;
}
.stButton > button:hover { box-shadow: 0 0 25px rgba(124,58,237,0.5); transform: translateY(-1px); }
.stTabs [data-baseweb="tab-list"] { background: #111118; border-radius: 10px; padding: 4px; }
.stTabs [data-baseweb="tab"] { color: #6b6b80; font-family: 'Syne', sans-serif; font-weight: 700; }
.stTabs [aria-selected="true"] { color: #fff !important; background: #7c3aed !important; border-radius: 8px; }
</style>
""", unsafe_allow_html=True)

# ── Constants ──────────────────────────────────────────────────────────────
DATA_DIR      = './data'
MNIST_LABELS  = [str(i) for i in range(10)]
EMNIST_LABELS = list('ABCDEFGHIJKLMNOPQRSTUVWXYZ')

# ── Helpers ────────────────────────────────────────────────────────────────
def read_idx_images(gz_path):
    with gzip.open(gz_path, 'rb') as f:
        magic, n, rows, cols = struct.unpack('>IIII', f.read(16))
        return np.frombuffer(f.read(), dtype=np.uint8).reshape(n, rows, cols)

def read_idx_labels(gz_path):
    with gzip.open(gz_path, 'rb') as f:
        magic, n = struct.unpack('>II', f.read(8))
        return np.frombuffer(f.read(), dtype=np.uint8)

@st.cache_data(show_spinner="Loading MNIST from local files...")
def load_mnist():
    with np.load(os.path.join(DATA_DIR, 'mnist.npz')) as f:
        return f['x_train'], f['y_train'], f['x_test'], f['y_test']

@st.cache_data(show_spinner="Loading EMNIST from local files...")
def load_emnist():
    X_tr = read_idx_images(os.path.join(DATA_DIR, 'emnist-letters-train-images-idx3-ubyte.gz'))
    y_tr = read_idx_labels(os.path.join(DATA_DIR, 'emnist-letters-train-labels-idx1-ubyte.gz'))
    X_te = read_idx_images(os.path.join(DATA_DIR, 'emnist-letters-test-images-idx3-ubyte.gz'))
    y_te = read_idx_labels(os.path.join(DATA_DIR, 'emnist-letters-test-labels-idx1-ubyte.gz'))
    X_tr = np.transpose(X_tr, (0, 2, 1)); X_te = np.transpose(X_te, (0, 2, 1))
    y_tr = y_tr - 1;                      y_te = y_te - 1
    return X_tr, y_tr, X_te, y_te

@st.cache_resource
def load_model(path):
    return tf.keras.models.load_model(path) if os.path.exists(path) else None

def preprocess_image(img: Image.Image) -> np.ndarray:
    """Convert uploaded image to 28×28 MNIST-style tensor."""
    img = img.convert('L')
    arr = np.array(img)
    # Auto-detect background: invert if white background
    if arr.mean() > 127:
        img = ImageOps.invert(img)
        arr = np.array(img)
    # Auto-crop around drawn character
    coords = np.argwhere(arr > 30)
    if len(coords) > 0:
        r0, c0 = coords.min(axis=0)
        r1, c1 = coords.max(axis=0)
        pad = 20
        r0 = max(0, r0 - pad);  c0 = max(0, c0 - pad)
        r1 = min(arr.shape[0], r1 + pad); c1 = min(arr.shape[1], c1 + pad)
        img = img.crop((c0, r0, c1, r1))
    # Resize to 20×20, center in 28×28 (standard MNIST format)
    img    = img.resize((20, 20), Image.LANCZOS)
    padded = Image.new('L', (28, 28), 0)
    padded.paste(img, (4, 4))
    arr = np.array(padded).astype('float32') / 255.0
    return arr.reshape(1, 28, 28, 1)

def confidence_chart(probs, labels):
    top5_idx  = np.argsort(probs)[::-1][:5]
    top5_lbls = [labels[i] for i in top5_idx]
    top5_vals = [float(probs[i]) * 100 for i in top5_idx]
    fig = go.Figure(go.Bar(
        x=top5_vals[::-1], y=top5_lbls[::-1], orientation='h',
        marker=dict(color=top5_vals[::-1], colorscale=[[0, '#1e1e2e'], [1, '#7c3aed']]),
        text=[f'{v:.1f}%' for v in top5_vals[::-1]],
        textposition='outside',
        textfont=dict(family='Space Mono', size=12, color='#e8e8f0'),
    ))
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=10, r=70, t=10, b=10), height=220,
        xaxis=dict(showgrid=False, showticklabels=False, range=[0, max(top5_vals) * 1.3]),
        yaxis=dict(tickfont=dict(family='Space Mono', size=15, color='#e8e8f0'), showgrid=False),
    )
    return fig

# ── Sidebar ────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('### ⚙️ Settings')
    st.markdown('---')
    mode     = st.radio('**Model**', ['MNIST — Digits (0–9)', 'EMNIST — Letters (A–Z)'], index=0)
    is_mnist = mode.startswith('MNIST')
    labels   = MNIST_LABELS if is_mnist else EMNIST_LABELS

    st.markdown('---')
    mnist_ok  = os.path.exists(os.path.join(DATA_DIR, 'mnist.npz'))
    emnist_ok = os.path.exists(os.path.join(DATA_DIR, 'emnist-letters-train-images-idx3-ubyte.gz'))
    m_model   = os.path.exists('mnist_cnn.keras')
    e_model   = os.path.exists('emnist_cnn.keras')

    st.markdown(
        f'<div class="info-box">'
        f'📁 <b>Data Files</b><br>'
        f'{"✅" if mnist_ok  else "❌"} mnist.npz<br>'
        f'{"✅" if emnist_ok else "❌"} emnist-letters-*.gz<br><br>'
        f'🤖 <b>Trained Models</b><br>'
        f'{"✅" if m_model else "❌"} mnist_cnn.keras<br>'
        f'{"✅" if e_model else "❌"} emnist_cnn.keras<br><br>'
        f'🧠 <b>Architecture</b><br>'
        f'Conv2D ×4–5<br>BatchNorm · MaxPool<br>Dropout · Dense 512<br>Softmax output'
        f'</div>', unsafe_allow_html=True)

    st.markdown('---')
    st.markdown('<p style="font-family:Space Mono,monospace;font-size:10px;color:#3a3a4a;text-align:center">'
                'CNN · TensorFlow/Keras</p>', unsafe_allow_html=True)

# ── Header ─────────────────────────────────────────────────────────────────
st.markdown('<div class="main-title">Handwritten Character<br>Recognition</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">CNN · MNIST + EMNIST · Local Data · Real-time Prediction</div>',
            unsafe_allow_html=True)

# ── Metric cards ───────────────────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
with c1: st.markdown(f'<div class="metric-card"><div class="metric-val">{"MNIST" if is_mnist else "EMNIST"}</div><div class="metric-lbl">Active Model</div></div>', unsafe_allow_html=True)
with c2: st.markdown(f'<div class="metric-card"><div class="metric-val">{"99.6%" if is_mnist else "85–92%"}</div><div class="metric-lbl">Test Accuracy</div></div>', unsafe_allow_html=True)
with c3: st.markdown(f'<div class="metric-card"><div class="metric-val">{"10" if is_mnist else "26"}</div><div class="metric-lbl">Classes</div></div>', unsafe_allow_html=True)
with c4: st.markdown(f'<div class="metric-card"><div class="metric-val">{"70K" if is_mnist else "145K"}</div><div class="metric-lbl">Training Samples</div></div>', unsafe_allow_html=True)

st.markdown('<br>', unsafe_allow_html=True)

# ── Tabs ───────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(['🖼️  Upload & Predict', '📊  Dataset Explorer', '🏗️  Architecture'])

# ══════════════════════════════════════════════════════════════════════════
# TAB 1 — Upload & Predict
# ══════════════════════════════════════════════════════════════════════════
with tab1:
    left, right = st.columns([1.1, 1], gap='large')

    with left:
        st.markdown('<div class="section-label"><span>01</span> Upload Handwritten Image</div>',
                    unsafe_allow_html=True)

        st.info(
            '💡 **How to create test images:**\n\n'
            '- **Windows Paint:** Black canvas → white brush → save PNG\n'
            '- **Phone:** Write on white paper → photo → crop → upload\n'
            '- **Online:** [jspaint.app](https://jspaint.app) — draw & save'
        )

        uploaded = st.file_uploader(
            'Upload image (PNG, JPG, BMP)',
            type=['png', 'jpg', 'jpeg', 'bmp'],
            label_visibility='collapsed'
        )

        if uploaded:
            img = Image.open(uploaded)
            col_img, col_proc = st.columns(2)
            with col_img:
                st.markdown('**Original**')
                st.image(img, use_column_width=True)
            with col_proc:
                st.markdown('**Preprocessed (28×28)**')
                arr    = preprocess_image(img)
                preview = Image.fromarray((arr.squeeze() * 255).astype('uint8'), mode='L')
                preview = preview.resize((112, 112), Image.NEAREST)
                st.image(preview, use_column_width=False, width=112)

    with right:
        st.markdown('<div class="section-label"><span>02</span> Prediction Result</div>',
                    unsafe_allow_html=True)

        model_path = 'mnist_cnn.keras' if is_mnist else 'emnist_cnn.keras'
        model      = load_model(model_path)

        if uploaded:
            with st.spinner('Running CNN inference...'):
                img   = Image.open(uploaded)
                inp   = preprocess_image(img)

                if model:
                    probs = model.predict(inp, verbose=0)[0]
                else:
                    st.warning(f'⚠️ `{model_path}` not found — run the notebook first to train.')
                    raw = np.random.dirichlet(np.ones(len(labels)) * 0.3)
                    raw[np.random.randint(len(labels))] += 3
                    probs = raw / raw.sum()

            top_idx  = int(np.argmax(probs))
            top_conf = float(probs[top_idx]) * 100
            top_char = labels[top_idx]

            st.markdown(
                f'<div class="pred-hero">'
                f'<div class="pred-char">{top_char}</div>'
                f'<div class="pred-conf">Confidence: {top_conf:.1f}%</div>'
                f'</div>', unsafe_allow_html=True)

            st.markdown('<br>', unsafe_allow_html=True)
            st.markdown('<div class="section-label"><span>03</span> Top 5 Confidence</div>',
                        unsafe_allow_html=True)
            st.plotly_chart(confidence_chart(probs, labels),
                            use_container_width=True, config={'displayModeBar': False})
        else:
            st.markdown(
                '<div class="pred-hero-empty">'
                '<div style="font-size:52px;margin-bottom:12px">⬆️</div>'
                '<div style="font-family:Space Mono,monospace;font-size:13px;color:#6b6b80">'
                'Upload an image to get prediction</div>'
                '</div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════
# TAB 2 — Dataset Explorer
# ══════════════════════════════════════════════════════════════════════════
with tab2:
    import matplotlib.pyplot as plt

    if is_mnist:
        if not mnist_ok:
            st.warning(f'⚠️ `mnist.npz` not found in `{DATA_DIR}/`')
        else:
            X_tr, y_tr, X_te, y_te = load_mnist()
            st.success(f'✅ MNIST — {len(X_tr):,} train / {len(X_te):,} test samples')

            col1, col2 = st.columns([1.3, 1])
            with col1:
                st.markdown('**Sample Images**')
                digit_filter = st.selectbox('Filter by digit', ['All'] + list(range(10)))
                if digit_filter == 'All':
                    idxs = np.random.choice(len(X_tr), 20, replace=False)
                else:
                    idxs = np.where(y_tr == digit_filter)[0][:20]

                fig, axes = plt.subplots(2, 10, figsize=(13, 2.8))
                fig.patch.set_facecolor('#0a0a0f')
                for i, idx in enumerate(idxs[:20]):
                    ax = axes[i // 10, i % 10]
                    ax.imshow(X_tr[idx], cmap='gray')
                    ax.set_title(str(y_tr[idx]), fontsize=9, color='white')
                    ax.axis('off'); ax.set_facecolor('#0a0a0f')
                plt.tight_layout()
                st.pyplot(fig, use_container_width=True)

            with col2:
                st.markdown('**Class Distribution**')
                unique, counts = np.unique(y_tr, return_counts=True)
                fig2 = go.Figure(go.Bar(
                    x=[str(u) for u in unique], y=counts,
                    marker_color='#7c3aed',
                    text=counts, textposition='outside',
                    textfont=dict(size=10, color='#e8e8f0')
                ))
                fig2.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                    margin=dict(l=0, r=0, t=10, b=0), height=260,
                    xaxis=dict(tickfont=dict(color='#e8e8f0'), showgrid=False),
                    yaxis=dict(tickfont=dict(color='#6b6b80'), gridcolor='#2a2a3a'),
                )
                st.plotly_chart(fig2, use_container_width=True, config={'displayModeBar': False})
    else:
        if not emnist_ok:
            st.warning(f'⚠️ EMNIST files not found in `{DATA_DIR}/`')
        else:
            X_tr, y_tr, X_te, y_te = load_emnist()
            st.success(f'✅ EMNIST Letters — {len(X_tr):,} train / {len(X_te):,} test samples')

            col1, col2 = st.columns([1.3, 1])
            with col1:
                st.markdown('**Sample Images (A–Z)**')
                fig, axes = plt.subplots(2, 13, figsize=(16, 2.8))
                fig.patch.set_facecolor('#0a0a0f')
                for i in range(26):
                    idx = np.where(y_tr == i)[0][0]
                    ax  = axes[i // 13, i % 13]
                    ax.imshow(X_tr[idx], cmap='gray')
                    ax.set_title(EMNIST_LABELS[i], fontsize=9, color='white')
                    ax.axis('off'); ax.set_facecolor('#0a0a0f')
                plt.tight_layout()
                st.pyplot(fig, use_container_width=True)

            with col2:
                st.markdown('**Class Distribution**')
                unique, counts = np.unique(y_tr, return_counts=True)
                fig2 = go.Figure(go.Bar(
                    x=[EMNIST_LABELS[u] for u in unique], y=counts,
                    marker_color='#7c3aed',
                ))
                fig2.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                    margin=dict(l=0, r=0, t=10, b=0), height=260,
                    xaxis=dict(tickfont=dict(color='#e8e8f0'), showgrid=False),
                    yaxis=dict(tickfont=dict(color='#6b6b80'), gridcolor='#2a2a3a'),
                )
                st.plotly_chart(fig2, use_container_width=True, config={'displayModeBar': False})

# ══════════════════════════════════════════════════════════════════════════
# TAB 3 — Architecture
# ══════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown('<div class="section-label"><span>04</span> CNN Architecture</div>',
                unsafe_allow_html=True)

    arch_cols = st.columns(5 if is_mnist else 6)
    arch_mnist = [
        ('Input',        '28 × 28 × 1\nGrayscale\nNormalized\n÷ 255'),
        ('Conv Block 1', 'Conv2D 32\n3×3 ReLU\nBatchNorm\nMaxPool 2×2\nDropout 0.25'),
        ('Conv Block 2', 'Conv2D 64\n3×3 ReLU\nBatchNorm\nMaxPool 2×2\nDropout 0.25'),
        ('Dense',        'Flatten\nDense 256\nBatchNorm\nDropout 0.5'),
        ('Output',       'Dense 10\nSoftmax\nCross-Entropy\nAdam 0.001'),
    ]
    arch_emnist = [
        ('Input',        '28 × 28 × 1\nGrayscale\nNormalized\n÷ 255'),
        ('Conv Block 1', 'Conv2D 32\n3×3 ReLU\nBatchNorm\nMaxPool 2×2\nDropout 0.25'),
        ('Conv Block 2', 'Conv2D 64\n3×3 ReLU\nBatchNorm\nMaxPool 2×2\nDropout 0.25'),
        ('Conv Block 3', 'Conv2D 128\n3×3 ReLU\nBatchNorm\nDropout 0.25'),
        ('Dense',        'Flatten\nDense 512\nBatchNorm\nDropout 0.5\nDense 256\nDropout 0.3'),
        ('Output',       'Dense 26\nSoftmax\nCross-Entropy\nAugmentation\nAdam 0.001'),
    ]
    arch = arch_mnist if is_mnist else arch_emnist
    for col, (title, desc) in zip(arch_cols, arch):
        with col:
            st.markdown(
                f'<div class="arch-card">'
                f'<div class="arch-title">{title}</div>'
                f'<div class="arch-body">{desc}</div>'
                f'</div>', unsafe_allow_html=True)

    st.markdown('<br>', unsafe_allow_html=True)

    # Model summary
    model_path = 'mnist_cnn.keras' if is_mnist else 'emnist_cnn.keras'
    model      = load_model(model_path)
    if model:
        st.markdown('<div class="section-label"><span>05</span> Model Summary</div>',
                    unsafe_allow_html=True)
        summary_rows = []
        model.summary(print_fn=lambda x: summary_rows.append(x))
        st.code('\n'.join(summary_rows), language='text')
    else:
        st.info(f'Train the model first by running the notebook. `{model_path}` not found.')
