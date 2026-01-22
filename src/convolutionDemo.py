import cv2
import matplotlib.pyplot as plt

example_cracked = "Decks/Cracked/7001-40.jpg"
example_uncracked = "Decks/Non-Cracked/7001-1.jpg"

def edge_comparison_plot(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0)
    sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1)
    sobel_combined = cv2.magnitude(sobelx, sobely)
    canny = cv2.Canny(img, 50, 150)

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.imshow(img, cmap="gray")
    plt.title("Original")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(sobel_combined, cmap="magma")
    plt.title("Sobel Edge")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(canny, cmap="hot")
    plt.title("Canny Edge")
    plt.axis("off")

    plt.suptitle("Edge Comparison (Sobel vs. Canny)", fontsize=14)
    plt.tight_layout()
    plt.show()

edge_comparison_plot(example_cracked)
edge_comparison_plot(example_uncracked)