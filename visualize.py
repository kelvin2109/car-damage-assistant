from matplotlib.patches import Polygon
import matplotlib.pyplot as plt
import cv2
import os
import math
import numpy as np



def draw_bboxes(image, bboxes, color, category_ids,thickness=2):
    """
    Draw bounding boxes on an image (in-place).

    Parameters
    ----------
    image : numpy.ndarray
        Image array (RGB format).
    bboxes : list or tuple
        Either a single bounding box [x, y, width, height] or a list of such boxes.
    color : tuple
        RGB color tuple (e.g., (0, 255, 0) for green).
    thickness : int
        Line thickness.

    Returns
    -------
    numpy.ndarray
        Image with bounding boxes drawn.
    """
    # Convert single bbox to list for uniform handling
    if isinstance(category_ids, list):
        for coor in bboxes:
            x, y, w, h = coor
            cv2.rectangle(image, (int(x), int(y)), (int(x + w), int(y + h)), color, thickness)
    else:
        x, y, w, h = bboxes
        cv2.rectangle(image, (int(x), int(y)), (int(x + w), int(y + h)), color, thickness)
    return image


def draw_masks(ax, segmentations, category_ids, class_color_map, alpha=0.5):
    """
    Draw segmentation masks as semi-transparent polygons on the given axes.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes to draw on.
    segmentations : list or array
        Either a single segmentation (flattened list of points) or a list of such.
    category_ids : int or list
        Either a single category ID or a list corresponding to each segmentation.
    class_color_map : dict
        Mapping from category_id to RGB color tuple (values from a colormap).
    alpha : float
        Transparency level for the mask fill.
    """
    # Convert scalar inputs to lists for uniform handling
    if not isinstance(segmentations, list):
        segmentations = [segmentations]
    if not isinstance(category_ids, list):
        category_ids = [category_ids]

    # Ensure both lists have the same length
    if len(segmentations) != len(category_ids):
        raise ValueError("segmentations and category_ids must have the same length")

    for cat_id, seg in zip(category_ids, segmentations):
        # COCO segmentation can be a list of points (flattened) or a list of polygons.
        # Here we assume a single polygon per object as a flattened list.
        points = np.array(seg).reshape(-1, 2).astype(np.int32)
        # Get base RGB color and add alpha channel
        base_color = class_color_map[cat_id][:3]  # ignore possible alpha from cmap
        facecolor = base_color + (alpha,)
        polygon = Polygon(points, facecolor=facecolor, edgecolor=None)
        ax.add_patch(polygon)


def coco_show_image(sample_img, img_dir="data/train/", num_class=None,
                    color=(0, 255, 0)):
    """
    Display sample images with bounding boxes and segmentation masks.

    Parameters
    ----------
    sample_img : pd.DataFrame
        DataFrame with columns ["file_name", "category_id", "segmentation", "bbox", "name"].
        "category_id" and "segmentation" can be lists for multiple objects per image.
    img_dir : str
        Directory containing the image files.
    num_class : int, optional
        Not used (kept for compatibility).
    color : tuple
        RGB color for bounding boxes (default green).
    figsize : tuple
        Figure size (width, height) for the plot.
    """
    # Convert to list of dictionaries for easier iteration
    records = sample_img[["file_name", "category_id", "segmentation", "bbox", "name"]].to_dict(orient="records")
    num_images = len(records)

    # Create a grid: 3 columns, rows as needed
    cols = 3
    rows = math.ceil(num_images / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(15,rows*4), constrained_layout=True)
    fig.suptitle("Sample Images", fontsize=16, y=1.02)

    # Flatten axes for easy indexing; handle single image case
    if num_images == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    # Determine unique class IDs and assign a distinct color from a colormap
    all_cat_ids = []
    for rec in records:
        if isinstance(rec["category_id"], list):
            all_cat_ids.extend(rec["category_id"])
        else:
            all_cat_ids.append(rec["category_id"])
    unique_classes = sorted(set(all_cat_ids))
    cmap = plt.get_cmap('tab10', len(unique_classes))
    class_color = {cls: cmap(i) for i, cls in enumerate(unique_classes)}

    # Process each image record
    for idx, rec in enumerate(records):
        if idx >= len(axes):
            break

        ax = axes[idx]
        img_path = os.path.join(img_dir, rec['file_name'])
        img = cv2.imread(img_path)
        if img is None:
            print(f"Warning: Could not read image {img_path}")
            ax.axis('off')
            continue

        # Convert BGR (OpenCV) to RGB (matplotlib)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Note: draw_bboxes and draw_masks now handle both scalar and list inputs,
        # so we can pass the data directly without converting to lists here.
        # However, we still need to separate the fields for clarity.

        # Draw bounding boxes on a copy of the image
        img_with_boxes = draw_bboxes(img_rgb.copy(), rec["bbox"], color, rec["category_id"], thickness=2)

        # Display the image
        ax.imshow(img_with_boxes)

        # Draw masks on the axes (semi-transparent overlays)
        draw_masks(ax, rec["segmentation"], rec["category_id"], class_color, alpha=0.5)

        # Set title with class name(s)
        if isinstance(rec["name"], list):
            # Show first two class names if there are many
            names = rec["name"]
            if len(names) > 2:
                title = f"Classes: {names[0]}, {names[1]} ..."
            else:
                title = f"Classes: {', '.join(names)}"
        else:
            title = f"Class: {rec['name']}"
        ax.set_title(title, fontsize=12)
        ax.axis('off')

    # Hide any unused subplots
    for j in range(idx + 1, len(axes)):
        axes[j].axis('off')

    plt.show()