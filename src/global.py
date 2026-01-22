import os
import pandas as pd
# import seaborn as sns
import matplotlib.pyplot as plt
from PIL import Image
from glob import glob
import matplotlib.patches as patches

# Load data
base_path = os.getcwd() #"/./"
categories = ["Decks", "Walls", "Pavements"]
labels = ["Cracked", "Non-cracked"]
data = []

for cat in categories:
    for label in labels:
        path = os.path.join(base_path, cat, label)
        count = len(glob(os.path.join(path, "*.jpg")))
        data.append((cat, label, count))

df = pd.DataFrame(data, columns=["Category", "Label", "ImageCount"])
df['Label_Category'] = df['Label'] + ' ' + df['Category']

# Enhanced Plot
df_plot = df.set_index('Label_Category')

colors = ['#FF5733', '#3357FF']
handles = [plt.Rectangle((0,0),1,1, color='#FF5733'), plt.Rectangle((0,0),1,1, color='#3357FF')]

df.plot(kind="bar", color=colors, x='Category', y='ImageCount', legend=True)
plt.title("Image Count per Category and Crack Status", fontsize=16, fontweight='bold')
plt.xlabel("Category", fontsize=12)
plt.ylabel("Number of Images", fontsize=12)
plt.xticks(rotation=30)
plt.legend(handles, labels, title="Crack Status")
plt.show()

def show_samples(base_path, category, label, n=5, grayscale=False):
    folder = os.path.join(base_path, category, label)
    image_paths = glob(os.path.join(folder, "*.jpg"))[:n]

    plt.figure(figsize=(4.2 * n, 5))
    
    for i, img_path in enumerate(image_paths):
        try:
            img = Image.open(img_path)
            if grayscale:
                img = img.convert("L")
            ax = plt.subplot(1, n, i + 1)
            ax.imshow(img, cmap='gray' if grayscale else None)
            ax.set_title(f"{label} #{i+1}", fontsize=12, fontweight='semibold', color='darkblue')
            ax.axis("off")

            # Optional: Add a border around each image
            rect = patches.Rectangle((0, 0), img.size[0], img.size[1], linewidth=2,
                                     edgecolor='royalblue', facecolor='none', transform=ax.transData)
            ax.add_patch(rect)

        except Exception as e:
            print(f"Error loading image: {img_path} -> {e}")
            continue

    plt.suptitle(f"{category.upper()} - {label} Samples", fontsize=18, fontweight='bold', color='darkred')
    plt.tight_layout(rect=[0, 0, 1, 0.90])
    plt.show()

# Show enhanced samples for all categories
for cat in categories:
    for label in labels:
        show_samples(base_path, cat, label, n=5)