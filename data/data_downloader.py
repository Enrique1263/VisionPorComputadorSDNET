import kagglehub

# Download latest version
path = kagglehub.dataset_download("data/structural-defects-network-concrete-crack-images")

print("Path to dataset files:", path)