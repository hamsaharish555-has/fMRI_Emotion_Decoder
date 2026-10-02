import torch
import pandas as pd
import numpy as np
from pathlib import Path

from gin_model import GINModel
GRAPH_DIR = Path("data/gnn_nyu107")
LABEL_FILE = Path("data/adhd200/NYU_test107.csv")
MODEL_FILE = Path("models/nyu107_5layer_gine.pt")
OUTPUT_FILE = Path("data/xai_nyu107_results.csv")
labels_df = pd.read_csv(LABEL_FILE)
labels_df["Subject"] = labels_df["ScanDir ID"].astype(int)
model = GINModel(
    input_dim=14,
    hidden_dim=64,
    num_classes=2
)

model.load_state_dict(
    torch.load(
        MODEL_FILE,
        weights_only=True
    )
)

model.eval()
try:
    from nilearn import datasets

    atlas = datasets.fetch_atlas_harvard_oxford(
        "cort-maxprob-thr25-2mm"
    )

    roi_names = list(atlas.labels)[1:]

except Exception as e:
    print("Could not load Harvard-Oxford atlas:")
    print(e)

    roi_names = [
        f"ROI_{i}"
        for i in range(1, 49)
    ]
if len(roi_names) != 48:
    print(
        "Warning: Expected 48 ROI names, found:",
        len(roi_names)
    )

    roi_names = roi_names[:48]

    while len(roi_names) < 48:
        roi_names.append(
            f"ROI_{len(roi_names) + 1}"
        )
        results = []

results = []
test_subjects = labels_df["Subject"].tolist()

print("Starting perturbation-based XAI...")
print("Test subjects:", len(test_subjects))
print("ROIs per subject: 48")
print()
for subject_id in test_subjects:

    graph_file = (
        GRAPH_DIR /
        f"{subject_id}_gnn.pt"
    )

    if not graph_file.exists():
        print(
            f"Skipping {subject_id}: graph not found"
        )
        continue
    data = torch.load(
        graph_file,
        weights_only=False
    )

    x = data.x.clone()
    edge_index = data.edge_index
    edge_attr = data.edge_attr

    batch = torch.zeros(
        x.size(0),
        dtype=torch.long
    )
    with torch.no_grad():

        original_output = model(
            x,
            edge_index,
            edge_attr,
            batch
        )

        probabilities = torch.softmax(
            original_output,
            dim=1
        )

        predicted_label = int(
            torch.argmax(
                probabilities,
                dim=1
            ).item()
        )

        original_score = float(
            probabilities[
                0,
                predicted_label
            ].item()
        )
    roi_importance = []

    for roi_index in range(x.size(0)):

        perturbed_x = x.clone()

        perturbed_x[roi_index] = 0

        with torch.no_grad():

            perturbed_output = model(
                perturbed_x,
                edge_index,
                edge_attr,
                batch
            )

            perturbed_probabilities = torch.softmax(
                perturbed_output,
                dim=1
            )

            perturbed_score = float(
                perturbed_probabilities[
                    0,
                    predicted_label
                ].item()
            )

        importance = abs(
            original_score -
            perturbed_score
        )

        roi_importance.append(
            importance
        )
    roi_importance = np.array(
        roi_importance
    )

    if np.all(
        roi_importance == 0
    ):
        print(
            f"Subject {subject_id}: "
            "No measurable ROI importance detected"
        )
        continue

    ranked_indices = np.argsort(
        roi_importance
    )[::-1]
    true_label = int(
        labels_df.loc[
            labels_df["Subject"] == subject_id,
            "Label"
        ].iloc[0]
    )

    for rank, roi_index in enumerate(
        ranked_indices[:10],
        start=1
    ):
        results.append(
            {
                "Subject": subject_id,
                "True_Label": true_label,
                "Predicted_Label": predicted_label,
                "Rank": rank,
                "ROI_Index": roi_index + 1,
                "ROI_Name": roi_names[roi_index],
                "Importance": float(
                    roi_importance[roi_index]
                )
            }
        )
    print(
        f"Subject {subject_id}: "
        f"True={true_label}, "
        f"Predicted={predicted_label}, "
        f"Top ROI={roi_names[ranked_indices[0]]}, "
        f"Importance={roi_importance[ranked_indices[0]]:.6f}"
    )


results_df = pd.DataFrame(
    results
)

results_df.to_csv(
    OUTPUT_FILE,
    index=False
)

print()
print("XAI COMPLETE")
print(
    "Total result rows:",
    len(results_df)
)
print(
    "Saved to:",
    OUTPUT_FILE
)