from utils import *
import sklearn
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import gc  # Garbage collection

def evaluate_image_similarity(image0, image1, n, center_filter, disp):
    torch.cuda.empty_cache()
    
    # Get the best rotation and matches
    _, rotated_image, _, _, _, _ = find_best_rotation_matches(image0, image1, n, disp)
    
    first = convert_cv_to_py(image0).to(device, non_blocking=True)
    second = convert_cv_to_py(rotated_image).to(device, non_blocking=True)

    # Extract the descriptors for the best matches
    feats0 = extractor.extract(first)
    feats1 = extractor.extract(second)

    # max_features = 11000
    # feats0 = {key: value[:, :max_features] if key in ["keypoints", "scales", "oris", "descriptors", "keypoint_scores"] else value for key, value in feats0.items()}
    # feats1 = {key: value[:, :max_features] if key in ["keypoints", "scales", "oris", "descriptors", "keypoint_scores"] else value for key, value in feats1.items()}

    matches01 = matcher({"image0": feats0, "image1": feats1})

    feats0, feats1, matches01 = [rbd(x) for x in [feats0, feats1, matches01]]  # Remove batch dimension


    _, _, matches = feats0["keypoints"], feats1["keypoints"], matches01["matches"]
    # logging.info(matches)
    # logging.info(f"matches shape = {matches.shape}")
    # logging.info(f'feats0 descriptors shape = {feats0["descriptors"].shape}')
    # logging.info(f'feats1 descriptors shape = {feats1["descriptors"].shape}')
    # # Extract matched descriptors using the indices from `matches`
    descriptors0 = feats0["descriptors"][matches[:, 0].long()]  # Indices from the first column of `matches`
    descriptors1 = feats1["descriptors"][matches[:, 1].long()]  # Indices from the second column of `matches`
    # logging.info(f'descriptors0 shape = {descriptors0.shape}')
    # logging.info(f'descriptors1 shape = {descriptors1.shape}')

    # Calculate cosine similarity between the descriptors of the best matches
    similarities = cosine_similarity(descriptors0.cpu().numpy(), descriptors1.cpu().numpy())

    # Average similarity of the matches
    avg_similarity = np.mean(similarities)

    logging.info(f"Average similarity (with rotation): {avg_similarity}")
    num_matches = len(matches)
    # Free memory by deleting variables that are no longer needed
    del first, second, feats0, feats1, descriptors0, descriptors1, matches01, matches
    gc.collect()  # Force garbage collection
    torch.cuda.empty_cache()

    return avg_similarity, num_matches




def compute_similarity_matrix(images, n, center_filter, disp):
    """
    Computes the similarity matrix for a list of images.
    
    Parameters:
    - images: List of images to compare.
    - n: Number of rotations to check.
    - center_filter: Unused parameter (could be for future filtering options).
    - disp: Boolean flag to display information.

    Returns:
    - similarity_matrix: A 2D NumPy array of similarity scores.
    """
    num_images = len(images)
    similarity_matrix = np.zeros((num_images, num_images), dtype=np.float32)

    for i in range(num_images):
        for j in range(num_images):
            logging.info(f"image {i} and {j}")
            _, similarity_matrix[i, j] = evaluate_image_similarity(images[i], images[j], n, center_filter, disp)

    return similarity_matrix

def save_similarity_heatmap(similarity_matrix, image_labels, filename="similarity_heatmap_nummatch.png"):
    """
    Saves a heatmap of image similarity scores as an image file.

    Parameters:
    - similarity_matrix: 2D NumPy array of similarity scores.
    - image_labels: List of labels for the images.
    - filename: Name of the file to save the heatmap.
    """
    plt.figure(figsize=(8, 6))
    sns.heatmap(similarity_matrix, annot=True, fmt=".2f", cmap="coolwarm", xticklabels=image_labels, yticklabels=image_labels)
    plt.title("Image Similarity Heatmap - Number of Matches")
    plt.xlabel("Image Index")
    plt.ylabel("Image Index")

    # Save the figure
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close()
    logging.info(f"Heatmap saved as {filename}")

# Load images
images = Path("output/images/evaluate")
image_paths = sorted(list(images.glob("*.jpg")))
image_list = []
i = 0
for path in image_paths: 
    i = i + 1
    image = cv2.imread(str(path))
    height, width, _ = image.shape
    logging.info(f"image {i} is [{height}, {width}]")
    reduced_image = cv2.resize(image, (width // 2, height // 2), interpolation=cv2.INTER_AREA)
    image_list.append(reduced_image)

labels = ["Version 1", "Version 2", "Final Version", "Galatsi Reference"]
n = 12  # Number of rotations to check
disp = False

# Compute similarity matrix
similarity_matrix = compute_similarity_matrix(image_list, n, center_filter=None, disp=disp)

# Save heatmap
save_similarity_heatmap(similarity_matrix, labels, "image_similarity.png")
