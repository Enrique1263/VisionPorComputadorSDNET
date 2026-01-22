import cv2
import os
from glob import glob
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Configuration
base_path = os.getcwd()
categories = ["Decks", "Walls", "Pavements"]
labels = ["Cracked", "Non-cracked"]

def extract_canny_features(image_path):
    """Extract Canny edge detection features from an image"""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    
    # Canny edge detection
    canny = cv2.Canny(img, 50, 150)
    
    # Calculate features
    canny_edge_density = np.sum(canny > 0) / canny.size
    canny_intensity = np.mean(canny)
    
    return {
        'canny_edge_density': canny_edge_density,
        'canny_intensity': canny_intensity
    }

def classify_image_canny(features, threshold=0.01):
    """Classify image as Cracked or Non-cracked based on Canny edge detection"""
    if features is None:
        return None
    
    # Images with higher edge density are more likely to be cracked
    prediction = features['canny_edge_density'] > threshold
    return "Cracked" if prediction else "Non-cracked"

# Process all images
results = []

for cat in categories:
    for label in labels:
        path = os.path.join(base_path, cat, label)
        image_paths = glob(os.path.join(path, "*.jpg"))
        
        for img_path in image_paths:
            features = extract_canny_features(img_path)
            if features is None:
                continue
            
            predicted_label = classify_image_canny(features)
            actual_label = label
            is_correct = predicted_label == actual_label
            
            results.append({
                'Category': cat,
                'Actual_Label': actual_label,
                'Predicted_Label': predicted_label,
                'Correct': is_correct,
                'Canny_Edge_Density': features['canny_edge_density'],
                'Canny_Intensity': features['canny_intensity']
            })

# Create DataFrame
df_results = pd.DataFrame(results)

# Generate Statistics
print("\n=== CANNY CLASSIFICATION RESULTS ===\n")
print(f"Total Images Processed: {len(df_results)}")
print(f"Correct Classifications: {df_results['Correct'].sum()}")
print(f"Overall Accuracy: {df_results['Correct'].mean() * 100:.2f}%\n")

# Accuracy per category
print("Accuracy per Category:")
for cat in categories:
    cat_df = df_results[df_results['Category'] == cat]
    accuracy = cat_df['Correct'].mean() * 100
    print(f"  {cat}: {accuracy:.2f}%")

# Accuracy per actual label
print("\nAccuracy per Actual Label:")
for label in labels:
    label_df = df_results[df_results['Actual_Label'] == label]
    accuracy = label_df['Correct'].mean() * 100
    print(f"  {label}: {accuracy:.2f}%")

# Confusion Matrix
print("\nConfusion Matrix:")
confusion = pd.crosstab(df_results['Actual_Label'], df_results['Predicted_Label'], margins=True)
print(confusion)

# Visualization 1: Accuracy by Category
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Accuracy comparison
accuracy_data = []
for cat in categories:
    cat_df = df_results[df_results['Category'] == cat]
    accuracy_data.append(cat_df['Correct'].mean() * 100)

axes[0].bar(categories, accuracy_data, color=['#FF5733', '#3357FF', '#33FF57'])
axes[0].set_title("Canny - Classification Accuracy per Category", fontsize=14, fontweight='bold')
axes[0].set_ylabel("Accuracy (%)", fontsize=12)
axes[0].set_ylim([0, 100])
axes[0].grid(axis='y', alpha=0.3)

# Confusion Matrix Heatmap
confusion_matrix = pd.crosstab(df_results['Actual_Label'], df_results['Predicted_Label'])
# Ensure both labels are present in the matrix
confusion_matrix = confusion_matrix.reindex(labels, fill_value=0).reindex(columns=labels, fill_value=0)

# Calculate percentages per actual class (row-wise)
confusion_matrix_pct = confusion_matrix.div(confusion_matrix.sum(axis=1), axis=0) * 100

im = axes[1].imshow(confusion_matrix_pct, cmap='Blues', aspect='auto', vmin=0, vmax=100)
axes[1].set_xticks([0, 1])
axes[1].set_yticks([0, 1])
axes[1].set_xticklabels(labels)
axes[1].set_yticklabels(labels)
axes[1].set_xlabel("Predicted Label", fontsize=12)
axes[1].set_ylabel("Actual Label", fontsize=12)
axes[1].set_title("Canny - Confusion Matrix (%)", fontsize=14, fontweight='bold')

# Add values to heatmap with counts and percentages
for i in range(len(labels)):
    for j in range(len(labels)):
        count = confusion_matrix.iloc[i, j]
        pct = confusion_matrix_pct.iloc[i, j]
        text = axes[1].text(j, i, f"{count}\n({pct:.1f}%)",
                          ha="center", va="center", color="black", fontweight='bold', fontsize=10)

plt.colorbar(im, ax=axes[1], label="Percentage (%)")
plt.tight_layout()
plt.show()

# Save results to CSV
df_results.to_csv("canny_classification_results.csv", index=False)
print("\nResults saved to 'canny_classification_results.csv'")