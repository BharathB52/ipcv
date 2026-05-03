import cv2
import numpy as np
import os

def load_reference_images(directory):
    """
    Scans the directory for all supported images (jpg, jpeg, png),
    computes SIFT features for each, and returns a list of dictionaries.
    """
    supported_extensions = ('.jpg', '.jpeg', '.png')
    # Pre-computation features - we keep this high for quality
    sift_library = cv2.SIFT_create(nfeatures=2000) 
    reference_data = []

    if not os.path.exists(directory):
        print(f"Error: Directory {directory} not found.")
        return []
        
    filenames = sorted([f for f in os.listdir(directory) if f.lower().endswith(supported_extensions)])
    
    if not filenames:
        print(f"Error: No supported images found in {directory}")
        return []

    print(f"Loading {len(filenames)} images from {directory}...")

    for filename in filenames:
        path = os.path.join(directory, filename)
        img_color = cv2.imread(path)
        if img_color is None:
            print(f"Warning: Could not read {path}. Skipping.")
            continue
            
        img_gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
        kp, des = sift_library.detectAndCompute(img_gray, None)
        
        if des is not None:
            reference_data.append({
                'filename': filename,
                'color_img': img_color,
                'kp': kp,
                'des': des,
                'h': img_gray.shape[0],
                'w': img_gray.shape[1]
            })
            print(f"  - Loaded {filename} ({len(kp)} features)")
        else:
            print(f"  - Warning: {filename} has no detectable SIFT features.")

    return reference_data

def main():
    directory = "photo_of_books"
    ref_data_list = load_reference_images(directory)
    
    if not ref_data_list:
        print("No valid reference images found. Exiting.")
        return

    # --- Feature Detection Setup (Optimized for speed) ---
    sift_live = cv2.SIFT_create(nfeatures=800) # Lower features = faster processing
    
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=40) # Slightly lower checks for speed
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    # ---------------------------------------------------

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    # --- Tracking State ---
    prev_H = None
    last_ref_idx = -1
    persistence_counter = 0
    MAX_PERSISTENCE = 5  # Reduced for zero-delay (faster removal)
    SMOOTHING_ALPHA = 0.8 # Higher = Snappier (Zero Delay). 1.0 = No filter.
    
    DETECTION_WIDTH = 640 
    # ----------------------

    print("\nStarting Zero-Delay AR.")
    print("- Snappiness: Enabled (Alpha 0.8)")
    print("- Smart Search: Enabled (Priority Image Caching)")
    print("- Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        h_full, w_full = frame.shape[:2]

        # Resolution Downscaling for Speed
        scale_factor = DETECTION_WIDTH / float(w_full)
        det_h = int(h_full * scale_factor)
        det_frame = cv2.resize(frame, (DETECTION_WIDTH, det_h))
        det_gray = cv2.cvtColor(det_frame, cv2.COLOR_BGR2GRAY)

        kp_frame, des_frame = sift_live.detectAndCompute(det_gray, None)

        found_match_this_frame = False
        current_match_idx = -1
        new_H = None

        if des_frame is not None and len(des_frame) > 10:
            # --- Priority Matching Optimization ---
            # If we saw an image in the previous frame, check it FIRST.
            # This skips redundant calculations for the rest of the folder.
            search_order = []
            if last_ref_idx != -1:
                search_order.append(last_ref_idx) # Check the "last success" first
            
            # Add all other indices
            for i in range(len(ref_data_list)):
                if i != last_ref_idx:
                    search_order.append(i)

            for idx in search_order:
                ref = ref_data_list[idx]
                try:
                    matches = flann.knnMatch(ref['des'], des_frame, k=2)
                    good_matches = [m for m, n in matches if m.distance < 0.7 * n.distance]
                    
                    if len(good_matches) > 12:
                        src_pts = np.float32([ref['kp'][m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                        dst_pts = np.float32([kp_frame[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2) / scale_factor

                        H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

                        if H is not None:
                            transformed_pts = cv2.perspectiveTransform(
                                np.float32([[0,0],[0,ref['h']],[ref['w'],ref['h']],[ref['w'],0]]).reshape(-1,1,2), H)
                            if cv2.contourArea(transformed_pts) > 1500:
                                new_H = H
                                current_match_idx = idx
                                found_match_this_frame = True
                                break # Exit image search once a match is found
                except: continue

        # --- Decision & Snappy Logic ---
        render_H = None
        render_ref_idx = -1

        if found_match_this_frame:
            # Snappy Smoothing (Higher Alpha = Faster response)
            if prev_H is not None and current_match_idx == last_ref_idx:
                # 80% new frame, 20% old frame (Zero Delay profile)
                render_H = SMOOTHING_ALPHA * new_H + (1.0 - SMOOTHING_ALPHA) * prev_H
            else:
                render_H = new_H
            
            prev_H = render_H
            last_ref_idx = current_match_idx
            render_ref_idx = current_match_idx
            persistence_counter = MAX_PERSISTENCE
        elif persistence_counter > 0 and prev_H is not None:
            render_H = prev_H
            render_ref_idx = last_ref_idx
            persistence_counter -= 1
        else:
            prev_H = None
            last_ref_idx = -1

        # --- High-Speed Rendering ---
        if render_H is not None:
            ref = ref_data_list[render_ref_idx]
            target_idx = (render_ref_idx + 1) % len(ref_data_list)
            overlay_source = ref_data_list[target_idx]['color_img']
            
            overlay_resized = cv2.resize(overlay_source, (ref['w'], ref['h']))
            warped_overlay = cv2.warpPerspective(overlay_resized, render_H, (w_full, h_full))

            mask_frame = np.zeros((h_full, w_full), dtype=np.uint8)
            corner_pts = np.float32([[0, 0], [0, ref['h']], [ref['w'], ref['h']], [ref['w'], 0]]).reshape(-1, 1, 2)
            dst_corners = cv2.perspectiveTransform(corner_pts, render_H)
            cv2.fillConvexPoly(mask_frame, np.int32(dst_corners), 255)

            # Feathering for quality, but slightly smaller kernel for speed
            mask_float = cv2.GaussianBlur(mask_frame, (5, 5), 0).astype(float) / 255.0
            mask_3ch = cv2.merge([mask_float, mask_float, mask_float])

            frame = (frame.astype(float) * (1.0 - mask_3ch) + warped_overlay.astype(float) * mask_3ch).astype(np.uint8)

        cv2.imshow('Zero-Delay AR', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
