import os
import tempfile

import numpy as np
import pandas as pd
import streamlit as st
import torch
import torch.nn.functional as F
import nibabel as nib
import matplotlib.pyplot as plt

from scipy.stats import skew, kurtosis

from nilearn import datasets
from nilearn.maskers import NiftiLabelsMasker
from nilearn.image import new_img_like

from torch_geometric.data import Data

from src.gin_model import GINModel


# ============================================================
# SETTINGS
# ============================================================

MODEL_PATH = "models/nyu107_5layer_gine.pt"

# Valid demo scan
DEMO_SCAN = "data/adhd200/sub-9750701_run-1_bold.nii.gz"

EDGE_THRESHOLD = 0.5

INPUT_DIM = 14
HIDDEN_DIM = 64
NUM_CLASSES = 2

DEVICE = torch.device("cpu")

CLASS_NAMES = {
    0: "Control",
    1: "ADHD"
}


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Explainable fMRI Brain Decoder",
    page_icon="🧠",
    layout="wide"
)


# ============================================================
# TITLE
# ============================================================

st.title("🧠 Explainable AI-Based fMRI Brain Decoder")

st.markdown(
    """
### ADHD Detection using fMRI + GINE Graph Neural Network

**Pipeline:**

fMRI Scan → Preprocessing → 48 Brain Regions → Brain Graph → 5-Layer GINE → Prediction → XAI
"""
)

st.warning(
    "Research prototype only. This system is not a clinical diagnostic tool."
)


# ============================================================
# SESSION STATE
# ============================================================

if "input_mode" not in st.session_state:
    st.session_state.input_mode = None

if "temp_path" not in st.session_state:
    st.session_state.temp_path = None

if "uploaded_filename" not in st.session_state:
    st.session_state.uploaded_filename = None


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():

    model = GINModel(
        input_dim=INPUT_DIM,
        hidden_dim=HIDDEN_DIM,
        num_classes=NUM_CLASSES
    )

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE,
        weights_only=False
    )

    if isinstance(checkpoint, dict):

        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]

        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]

        else:
            state_dict = checkpoint

    else:
        state_dict = checkpoint

    model.load_state_dict(
        state_dict,
        strict=False
    )

    model.to(DEVICE)
    model.eval()

    return model


# ============================================================
# LOAD HARVARD-OXFORD ATLAS
# ============================================================

@st.cache_resource
def load_atlas():

    atlas = datasets.fetch_atlas_harvard_oxford(
        "cort-maxprob-thr25-2mm"
    )

    atlas_img = atlas.maps

    labels = list(atlas.labels)

    return atlas_img, labels


# ============================================================
# EXTRACT ROI TIME SERIES
# ============================================================

def extract_roi_timeseries(scan_path, atlas_img):

    masker = NiftiLabelsMasker(
        labels_img=atlas_img,
        standardize=False,
        detrend=False,
        verbose=0
    )

    roi_timeseries = masker.fit_transform(scan_path)

    return roi_timeseries


# ============================================================
# CREATE NODE FEATURES
# ============================================================

def create_node_features(roi_timeseries):

    num_rois = roi_timeseries.shape[1]

    features = []

    # Correlation between brain regions
    corr = np.corrcoef(
        roi_timeseries.T
    )

    corr = np.nan_to_num(
        corr,
        nan=0.0,
        posinf=0.0,
        neginf=0.0
    )

    for i in range(num_rois):

        signal = roi_timeseries[:, i]

        signal = np.nan_to_num(
            signal,
            nan=0.0,
            posinf=0.0,
            neginf=0.0
        )

        # Basic statistics
        mean_value = np.mean(signal)

        std_value = np.std(signal)

        min_value = np.min(signal)

        max_value = np.max(signal)

        median_value = np.median(signal)

        skew_value = skew(signal)

        kurtosis_value = kurtosis(signal)

        mean_square = np.mean(
            np.square(signal)
        )

        # Correlation values
        row_corr = corr[i]

        # Remove self-correlation
        row_corr_without_self = np.delete(
            row_corr,
            i
        )

        mean_corr = np.mean(
            row_corr_without_self
        )

        mean_abs_corr = np.mean(
            np.abs(row_corr_without_self)
        )

        sum_abs_corr = np.sum(
            np.abs(row_corr_without_self)
        )

        count_abs_corr = np.sum(
            np.abs(row_corr_without_self) > 0.3
        )

        positive_corr = row_corr_without_self[
            row_corr_without_self > 0
        ]

        negative_corr = row_corr_without_self[
            row_corr_without_self < 0
        ]

        if len(positive_corr) > 0:
            mean_positive_corr = np.mean(
                positive_corr
            )
        else:
            mean_positive_corr = 0.0

        if len(negative_corr) > 0:
            mean_negative_corr = np.mean(
                negative_corr
            )
        else:
            mean_negative_corr = 0.0

        node_feature = [
            mean_value,
            std_value,
            min_value,
            max_value,
            median_value,
            skew_value,
            kurtosis_value,
            mean_square,
            mean_corr,
            mean_abs_corr,
            sum_abs_corr,
            count_abs_corr,
            mean_positive_corr,
            mean_negative_corr
        ]

        features.append(
            node_feature
        )

    features = np.asarray(
        features,
        dtype=np.float32
    )

    features = np.nan_to_num(
        features,
        nan=0.0,
        posinf=0.0,
        neginf=0.0
    )

    return features, corr


# ============================================================
# CREATE BRAIN GRAPH
# ============================================================

def create_graph(node_features, correlation):

    edge_sources = []
    edge_targets = []
    edge_values = []

    num_nodes = correlation.shape[0]

    for i in range(num_nodes):

        for j in range(num_nodes):

            if i == j:
                continue

            value = correlation[i, j]

            if abs(value) > EDGE_THRESHOLD:

                edge_sources.append(i)

                edge_targets.append(j)

                edge_values.append(value)

    # Avoid completely empty graph
    if len(edge_sources) == 0:

        for i in range(num_nodes):

            for j in range(num_nodes):

                if i != j:

                    edge_sources.append(i)
                    edge_targets.append(j)
                    edge_values.append(
                        correlation[i, j]
                    )

    edge_index = torch.tensor(
        [
            edge_sources,
            edge_targets
        ],
        dtype=torch.long
    )

    edge_attr = torch.tensor(
        edge_values,
        dtype=torch.float32
    ).view(-1, 1)

    x = torch.tensor(
        node_features,
        dtype=torch.float32
    )

    graph = Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr
    )

    return graph


# ============================================================
# PREDICTION
# ============================================================

def predict(model, graph):

    graph = graph.to(DEVICE)

    graph.batch = torch.zeros(
        graph.x.shape[0],
        dtype=torch.long,
        device=DEVICE
    )

    with torch.no_grad():

        output = model(
            graph.x,
            graph.edge_index,
            graph.edge_attr,
            graph.batch
        )

        probabilities = F.softmax(
            output,
            dim=1
        )

        predicted_class = torch.argmax(
            probabilities,
            dim=1
        ).item()

        confidence = probabilities[
            0,
            predicted_class
        ].item()

    return (
        predicted_class,
        confidence,
        probabilities.cpu().numpy()[0]
    )


# ============================================================
# XAI - ROI PERTURBATION
# ============================================================

def calculate_roi_importance(
    model,
    graph,
    predicted_class
):

    base_graph = graph.clone()

    base_graph = base_graph.to(DEVICE)

    base_graph.batch = torch.zeros(
        base_graph.x.shape[0],
        dtype=torch.long,
        device=DEVICE
    )

    with torch.no_grad():

        base_output = model(
            base_graph.x,
            base_graph.edge_index,
            base_graph.edge_attr,
            base_graph.batch
        )

        base_probability = F.softmax(
            base_output,
            dim=1
        )[0, predicted_class].item()

    importance = []

    for roi_index in range(
        graph.x.shape[0]
    ):

        perturbed_graph = graph.clone()

        # Save original node
        original_node = (
            perturbed_graph.x[
                roi_index
            ].clone()
        )

        # Remove node information
        perturbed_graph.x[
            roi_index
        ] = 0

        perturbed_graph = perturbed_graph.to(
            DEVICE
        )

        perturbed_graph.batch = torch.zeros(
            perturbed_graph.x.shape[0],
            dtype=torch.long,
            device=DEVICE
        )

        with torch.no_grad():

            output = model(
                perturbed_graph.x,
                perturbed_graph.edge_index,
                perturbed_graph.edge_attr,
                perturbed_graph.batch
            )

            probability = F.softmax(
                output,
                dim=1
            )[0, predicted_class].item()

        change = abs(
            base_probability - probability
        )

        importance.append(
            change
        )

        # Restore node
        perturbed_graph.x[
            roi_index
        ] = original_node

    return np.asarray(
        importance,
        dtype=np.float32
    )


# ============================================================
# SAFE NIFTI VALIDATION
# ============================================================

def validate_fmri_file(path):

    try:

        image = nib.load(path)

        shape = image.shape

        if len(shape) != 4:

            return False, (
                f"Expected 4D fMRI scan, "
                f"but received shape {shape}"
            )

        return True, shape

    except Exception as e:

        return False, str(e)


# ============================================================
# SIMPLE BRAIN VISUALIZATION
# ============================================================

def display_brain_image(scan_path):

    try:

        image = nib.load(scan_path)

        data = image.get_fdata()

        # Average across time
        mean_data = np.mean(
            data,
            axis=3
        )

        # Replace invalid values
        mean_data = np.nan_to_num(
            mean_data,
            nan=0.0,
            posinf=0.0,
            neginf=0.0
        )

        # Middle slices
        x = mean_data.shape[0] // 2
        y = mean_data.shape[1] // 2
        z = mean_data.shape[2] // 2

        fig, axes = plt.subplots(
            1,
            3,
            figsize=(12, 4)
        )

        axes[0].imshow(
            np.rot90(mean_data[x, :, :]),
            cmap="gray"
        )

        axes[0].set_title(
            "Sagittal"
        )

        axes[1].imshow(
            np.rot90(mean_data[:, y, :]),
            cmap="gray"
        )

        axes[1].set_title(
            "Coronal"
        )

        axes[2].imshow(
            np.rot90(mean_data[:, :, z]),
            cmap="gray"
        )

        axes[2].set_title(
            "Axial"
        )

        for ax in axes:
            ax.axis("off")

        fig.suptitle(
            "Mean fMRI Brain Image",
            fontsize=16
        )

        plt.tight_layout()

        st.pyplot(
            fig,
            clear_figure=True
        )

        plt.close(fig)

        return True

    except Exception as e:

        st.warning(
            f"Brain visualization could not be displayed: {e}"
        )

        return False


# ============================================================
# XAI BRAIN VISUALIZATION
# ============================================================

def display_xai_brain(
    atlas_img,
    importance
):

    try:

        atlas_image = nib.load(
            atlas_img
        )

        atlas_data = atlas_image.get_fdata()

        xai_data = np.zeros_like(
            atlas_data,
            dtype=np.float32
        )

        num_labels = len(importance)

        for roi_index in range(
            1,
            min(
                num_labels + 1,
                int(atlas_data.max()) + 1
            )
        ):

            xai_data[
                atlas_data == roi_index
            ] = importance[
                roi_index - 1
            ]

        xai_image = new_img_like(
            atlas_image,
            xai_data
        )

        # Middle slices
        x = xai_data.shape[0] // 2
        y = xai_data.shape[1] // 2
        z = xai_data.shape[2] // 2

        fig, axes = plt.subplots(
            1,
            3,
            figsize=(12, 4)
        )

        vmax = np.max(
            importance
        )

        if vmax <= 0:
            vmax = 1.0

        axes[0].imshow(
            np.rot90(
                xai_data[x, :, :]
            ),
            cmap="hot",
            vmin=0,
            vmax=vmax
        )

        axes[0].set_title(
            "Sagittal XAI"
        )

        axes[1].imshow(
            np.rot90(
                xai_data[:, y, :]
            ),
            cmap="hot",
            vmin=0,
            vmax=vmax
        )

        axes[1].set_title(
            "Coronal XAI"
        )

        axes[2].imshow(
            np.rot90(
                xai_data[:, :, z]
            ),
            cmap="hot",
            vmin=0,
            vmax=vmax
        )

        axes[2].set_title(
            "Axial XAI"
        )

        for ax in axes:
            ax.axis("off")

        fig.suptitle(
            "XAI Brain Region Importance",
            fontsize=16
        )

        plt.tight_layout()

        st.pyplot(
            fig,
            clear_figure=True
        )

        plt.close(fig)

        return xai_image

    except Exception as e:

        st.warning(
            f"XAI brain visualization could not be displayed: {e}"
        )

        return None


# ============================================================
# INPUT SECTION
# ============================================================

st.header("📂 Input fMRI Scan")

st.write(
    "Choose a demo scan or upload your own valid 4D NIfTI fMRI scan."
)

col1, col2 = st.columns(2)

with col1:

    use_demo = st.button(
        "🧠 Use Demo fMRI Scan",
        use_container_width=True
    )

with col2:

    uploaded_file = st.file_uploader(
        "Upload fMRI (.nii or .nii.gz)",
        type=[
            "nii",
            "gz"
        ]
    )


# ============================================================
# SELECT INPUT
# ============================================================

if use_demo:

    if os.path.exists(DEMO_SCAN):

        st.session_state.input_mode = "demo"

        st.session_state.temp_path = DEMO_SCAN

        st.session_state.uploaded_filename = None

        st.success(
            "Demo scan selected: "
            "sub-9750701_run-1_bold.nii.gz"
        )

    else:

        st.error(
            "Demo scan was not found."
        )


elif uploaded_file is not None:

    try:

        st.session_state.input_mode = "upload"

        st.session_state.uploaded_filename = (
            uploaded_file.name
        )

        suffix = ".nii.gz"

        if uploaded_file.name.endswith(".nii"):
            suffix = ".nii"

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=suffix
        ) as tmp:

            tmp.write(
                uploaded_file.getbuffer()
            )

            st.session_state.temp_path = (
                tmp.name
            )

        st.success(
            f"Uploaded: {uploaded_file.name}"
        )

    except Exception as e:

        st.error(
            f"Upload failed: {e}"
        )


# ============================================================
# CURRENT INPUT
# ============================================================

scan_path = st.session_state.temp_path


if scan_path is not None:

    st.divider()

    st.header(
        "🔬 fMRI Analysis"
    )

    # ========================================================
    # STEP 1
    # ========================================================

    st.subheader(
        "### Step 1/6 — Reading fMRI scan..."
    )

    valid, result = validate_fmri_file(
        scan_path
    )

    if not valid:

        st.error(
            f"Invalid fMRI file: {result}"
        )

        st.stop()

    scan_shape = result

    file_size_mb = (
        os.path.getsize(scan_path)
        / (1024 * 1024)
    )

    st.info(
        f"File size: {file_size_mb:.2f} MB"
    )

    st.success(
        f"Valid 4D fMRI scan: {scan_shape}"
    )


    # ========================================================
    # BRAIN IMAGE
    # ========================================================

    st.subheader(
        "### 🧠 Brain Scan Visualization"
    )

    st.write(
        "The image below shows the mean fMRI signal across all time points."
    )

    display_brain_image(
        scan_path
    )


    # ========================================================
    # LOAD MODEL + ATLAS
    # ========================================================

    st.subheader(
        "### Step 2/6 — Loading AI model..."
    )

    try:

        model = load_model()

        atlas_img, atlas_labels = (
            load_atlas()
        )

        st.success(
            "GINE model and Harvard-Oxford atlas loaded."
        )

    except Exception as e:

        st.error(
            f"Model/atlas loading failed: {e}"
        )

        st.stop()


    # ========================================================
    # STEP 3
    # ========================================================

    st.subheader(
        "### Step 3/6 — Extracting 48 brain regions..."
    )

    try:

        roi_timeseries = (
            extract_roi_timeseries(
                scan_path,
                atlas_img
            )
        )

        st.success(
            f"ROI extraction completed: "
            f"{roi_timeseries.shape}"
        )

        st.write(
            f"Number of brain regions: "
            f"**{roi_timeseries.shape[1]}**"
        )

        st.write(
            f"Number of time points: "
            f"**{roi_timeseries.shape[0]}**"
        )

    except Exception as e:

        st.error(
            f"ROI extraction failed: {e}"
        )

        st.stop()


    # ========================================================
    # STEP 4
    # ========================================================

    st.subheader(
        "### Step 4/6 — Creating brain graph..."
    )

    try:

        node_features, correlation = (
            create_node_features(
                roi_timeseries
            )
        )

        graph = create_graph(
            node_features,
            correlation
        )

        st.success(
            "Brain graph created successfully."
        )

        st.write(
            f"Nodes: **{graph.x.shape[0]}**"
        )

        st.write(
            f"Edges: **{graph.edge_index.shape[1]}**"
        )

        st.write(
            f"Node features: **{graph.x.shape[1]}**"
        )

    except Exception as e:

        st.error(
            f"Graph construction failed: {e}"
        )

        st.stop()


    # ========================================================
    # STEP 5
    # ========================================================

    st.subheader(
        "### Step 5/6 — GINE prediction..."
    )

    try:

        predicted_class, confidence, probabilities = (
            predict(
                model,
                graph
            )
        )

        predicted_label = CLASS_NAMES[
            predicted_class
        ]

        st.success(
            "Prediction completed."
        )

        result_col1, result_col2 = st.columns(2)

        with result_col1:

            st.metric(
                "Prediction",
                predicted_label
            )

        with result_col2:

            st.metric(
                "Confidence",
                f"{confidence * 100:.2f}%"
            )

        st.write(
            "### Prediction probabilities"
        )

        probability_df = pd.DataFrame(
            {
                "Class": [
                    "Control",
                    "ADHD"
                ],
                "Probability": [
                    probabilities[0],
                    probabilities[1]
                ]
            }
        )

        probability_df[
            "Probability"
        ] = (
            probability_df[
                "Probability"
            ] * 100
        )

        st.bar_chart(
            probability_df.set_index(
                "Class"
            )
        )

    except Exception as e:

        st.error(
            f"Prediction failed: {e}"
        )

        st.stop()


    # ========================================================
    # STEP 6
    # ========================================================

    st.subheader(
        "### Step 6/6 — Explainable AI analysis..."
    )

    try:

        importance = calculate_roi_importance(
            model,
            graph,
            predicted_class
        )

        st.success(
            "Perturbation-based XAI completed."
        )

        st.write(
            """
            The XAI method temporarily removes the information
            from each brain region and measures how much the
            model's predicted probability changes.
            """
        )

    except Exception as e:

        st.error(
            f"XAI calculation failed: {e}"
        )

        st.stop()


    # ========================================================
    # ROI NAMES
    # ========================================================

    # Harvard-Oxford has background as first label.
    if len(atlas_labels) >= len(importance) + 1:

        roi_names = atlas_labels[
            1:len(importance) + 1
        ]

    else:

        roi_names = [
            f"ROI {i + 1}"
            for i in range(
                len(importance)
            )
        ]


    # ========================================================
    # XAI TABLE
    # ========================================================

    xai_df = pd.DataFrame(
        {
            "ROI": roi_names,
            "Importance": importance
        }
    )

    xai_df = xai_df.sort_values(
        "Importance",
        ascending=False
    ).reset_index(
        drop=True
    )

    xai_df[
        "Rank"
    ] = np.arange(
        1,
        len(xai_df) + 1
    )

    # ========================================================
    # TOP 10
    # ========================================================

    st.header(
        "🧠 Top 10 Important Brain Regions"
    )

    top10 = xai_df.head(10)

    st.dataframe(
        top10[
            [
                "Rank",
                "ROI",
                "Importance"
            ]
        ],
        use_container_width=True,
        hide_index=True
    )


    # ========================================================
    # TOP 10 BAR CHART
    # ========================================================

    st.subheader(
        "📊 ROI Importance"
    )

    chart_df = top10[
        [
            "ROI",
            "Importance"
        ]
    ].copy()

    chart_df = chart_df.set_index(
        "ROI"
    )

    st.bar_chart(
        chart_df
    )


    # ========================================================
    # XAI BRAIN IMAGE
    # ========================================================

    st.header(
        "🔥 XAI Brain Visualization"
    )

    st.write(
        """
        Brighter regions represent brain regions where
        perturbing the ROI produced a larger change in the
        model's prediction for this scan.
        """
    )

    display_xai_brain(
        atlas_img,
        importance
    )


    # ========================================================
    # ALL ROI RESULTS
    # ========================================================

    st.header(
        "📋 All 48 Brain Regions"
    )

    st.dataframe(
        xai_df[
            [
                "Rank",
                "ROI",
                "Importance"
            ]
        ],
        use_container_width=True,
        hide_index=True
    )


    # ========================================================
    # DOWNLOAD XAI RESULTS
    # ========================================================

    csv_data = xai_df.to_csv(
        index=False
    )

    st.download_button(
        label="⬇️ Download XAI Results CSV",
        data=csv_data,
        file_name="xai_roi_importance.csv",
        mime="text/csv",
        use_container_width=True
    )


    # ========================================================
    # PIPELINE SUMMARY
    # ========================================================

    st.header(
        "🔄 Complete Pipeline"
    )

    pipeline_col1, pipeline_col2 = st.columns(2)

    with pipeline_col1:

        st.markdown(
            """
            **1. fMRI Input**

            ↓

            **2. Preprocessing**

            ↓

            **3. Harvard-Oxford Atlas**

            ↓

            **4. 48 Brain Regions**

            ↓

            **5. Node Feature Extraction**
            """
        )

    with pipeline_col2:

        st.markdown(
            """
            **6. Brain Graph**

            ↓

            **7. 5-Layer GINE**

            ↓

            **8. ADHD / Control Prediction**

            ↓

            **9. Confidence**

            ↓

            **10. Perturbation XAI**
            """
        )


    # ========================================================
    # TECHNICAL DETAILS
    # ========================================================

    with st.expander(
        "🔧 Technical Details"
    ):

        st.write(
            f"""
            **Model:** 5-Layer GINE

            **Input dimension:** {INPUT_DIM}

            **Hidden dimension:** {HIDDEN_DIM}

            **Number of classes:** {NUM_CLASSES}

            **ROI count:** {graph.x.shape[0]}

            **Node feature count:** {graph.x.shape[1]}

            **Graph edges:** {graph.edge_index.shape[1]}

            **Edge threshold:** {EDGE_THRESHOLD}

            **Device:** CPU

            **Atlas:** Harvard-Oxford cortical atlas

            **XAI:** ROI perturbation / occlusion analysis
            """
        )


    # ========================================================
    # DISCLAIMER
    # ========================================================

    st.divider()

    st.caption(
        """
        ⚠️ This application is a research/academic prototype.
        The prediction and XAI visualization are not medical diagnoses,
        and the highlighted brain regions should not be interpreted
        as confirmed ADHD biomarkers or causal findings.
        """
    )