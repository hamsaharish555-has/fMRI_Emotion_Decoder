from nilearn.datasets import fetch_adhd

print("Downloading ADHD fMRI dataset...")

dataset = fetch_adhd(
    n_subjects=5,
    data_dir="data/adhd"
)

print("Download complete!")
print("Number of fMRI scans:", len(dataset.func))
print("First scan:", dataset.func[0])
