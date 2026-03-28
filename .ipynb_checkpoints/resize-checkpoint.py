import cv2
import numpy as np

class resize():
        
    """
    Resize image to maintain aspect ratio while constraining dimensions:
    - Shorter side is resized to short_target (default: 800px)
    - Longer side does not exceed long_target (default: 1333px 
    
    The image is scaled by a single factor to meet both constraints:
    1. Scale factor = short_target / current_shorter_side
    2. If resulting longer_side > long_target, use long_target / current_longer_side
    
    This ensures aspect ratio preservation while bounding to maximum dimension.

    
    """
    
    def __init__(self,min_size = 800, max_size = 1333):
        self.min_size = min_size
        self.max_size = max_size
        

    def polygon_to_mask(self, segmentation, height, width):
        mask = np.zeros((height, width), dtype=np.uint8)
    
        for poly in segmentation:
            pts = np.array(poly).reshape(-1, 2).astype(np.int32)
            cv2.fillPoly(mask, [pts], 1)
    
        return mask

    def mask_to_polygon(self, mask, min_area=10):

        mask = mask.astype(np.uint8)
    
        contours, _ = cv2.findContours(
            mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )
    
        polygons = []
    
        for contour in contours:
            if cv2.contourArea(contour) < min_area:
                continue
    
            contour = contour.squeeze()
    
            if len(contour.shape) != 2:
                continue
    
            poly = contour.flatten().astype(float).tolist()
            polygons.append(poly)
    
        return polygons


    def __call__(self, image, boxes, mask=None):
        """
        image: np.ndarray(H, W, C)
        boxes: np.ndarray(N,4) in xyxy format
        """
        h,w = image.shape[:2]

        shorter = min(h, w)
        longer = max(h, w)

        scale = self.min_size / shorter
        if longer * scale > self.max_size:
            scale = self.max_size / longer

        new_h = int(round(h * scale))
        new_w = int(round(w * scale))

        image_resized = cv2.resize(image,
                                    (new_w,new_h), 
                                    interpolation=cv2.INTER_LINEAR
                                   )
        
        boxes_resized = boxes.copy().astype(np.float32)
        boxes_resized *= scale

        # resize mask (if exists)
        if mask is not None:
            mask = self.polygon_to_mask(mask, h, w)
            
            mask_resized = cv2.resize(
                mask,
                (new_w, new_h),
                interpolation=cv2.INTER_NEAREST  # VERY IMPORTANT
            )
            poly_mask = self.mask_to_polygon(mask_resized)
            
        else:
            mask_resized = None

        return image_resized, boxes_resized, poly_mask, scale