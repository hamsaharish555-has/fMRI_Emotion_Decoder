from nilearn import datasets
from nilearn.maskers import NiftiLabelsMasker


def extract_roi_signals(fmri_path):
    """Extract brain-region signals from an fMRI scan."""

    # Load Harvard-Oxford cortical atlas
    atlas = datasets.fetch_atlas_harvard_oxford(
        "cort-maxprob-thr25-2mm"
    )

    # Create ROI masker
    masker = NiftiLabelsMasker(
        labels_img=atlas.maps,
        standardize="zscore_sample"
    )

    # Extract ROI signals
    signals = masker.fit_transform(fmri_path)

    return signals


if __name__ == "__main__":
    fmri_path = (
        "data/adhd/adhd/data/0010042/"
        "0010042_rest_tshift_RPI_voreg_mni.nii.gz"
    )

    signals = extract_roi_signals(fmri_path)

    print("Preprocessing successful!")
    print("ROI signal shape:", signals.shape)
