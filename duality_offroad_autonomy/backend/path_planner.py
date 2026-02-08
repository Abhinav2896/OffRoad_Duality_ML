import numpy as np
import cv2
import heapq
import time

class PathPlanner:
    def __init__(self):
        # Class IDs
        self.SKY_CLASS = 0
        self.DRIVABLE_CLASSES = {1, 2}  # Landscape, Dry Grass
        self.GROUND_CLUTTER_CLASS = 3
        self.ROCKS_CLASS = 4
        self.LOGS_CLASS = 5
        self.BUSHES_CLASS = 6
        self.TREES_CLASS = 7

        # Horizon ratio tuned to allow >=65% path length
        self.HORIZON_RATIO = 0.35

        # Cost weights (lower is better)
        self.CLASS_COST = {
            self.SKY_CLASS: np.inf,        # Masked by horizon guard; treat as unreachable
            1: 1.0,                        # Landscape
            2: 1.05,                       # Dry Grass (Reduced: soft obstacle)
            3: 1.1,                        # Ground Clutter (Reduced: minor roughness)
            6: 1.3,                        # Bushes (Reduced: drive through if needed)
            7: 4.0,                        # Trees
            5: 20.0,                       # Logs (Hard obstacle: avoid strictly)
            4: 25.0,                       # Rocks (Hard obstacle: avoid strictly)
        }

        # Hard obstacles strongly block path
        self.HARD_OBSTACLES = {self.ROCKS_CLASS, self.LOGS_CLASS}

        # Shadow detection thresholds
        self.SHADOW_V_FRAC = 0.35
        self.SHADOW_S_FRAC = 0.35
        self.SHADOW_GRAD_THRESH = 12.0

        # Morphological cleanup
        self.OPEN_KERNEL_SIZE = 5         # Increased to 5 to remove larger noise speckles
        self.MIN_OBSTACLE_AREA_FRAC = 0.001  # Increased to 0.1% to ignore desert texture noise

        # Perspective weighting
        self.UPPER_THIRD_WEIGHT = 0.5   # reduce non-hard obstacle penalties near horizon
        self.MIDDLE_WEIGHT = 0.8
        self.NEAR_WEIGHT = 1.0

        # Path requirements
        self.MIN_PATH_LENGTH_FRAC = 0.75  # Target 75% of drivable height


        # Smoothing
        self.SMOOTHING_WINDOW_SIZE = 15
        
        # Planning performance limits
        self.SCALE_FACTOR = 0.50      # Increased to 0.5 for better gap resolution
        self.MAX_ITERATIONS = 400000  # Increased for higher resolution
        self.TIME_LIMIT_SEC = 1.50    # Increased time limit

    def _detect_shadows(self, image_rgb):
        """
        Returns a boolean mask of likely shadow regions using HSV and local contrast.
        """
        hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
        h, s, v = cv2.split(hsv)

        v_thresh = int(np.quantile(v, self.SHADOW_V_FRAC))
        s_thresh = int(np.quantile(s, self.SHADOW_S_FRAC))

        # Gradient magnitude for local contrast
        gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        grad_mag = cv2.magnitude(gx, gy)

        low_brightness = v < v_thresh
        low_saturation = s < s_thresh
        low_contrast = grad_mag < self.SHADOW_GRAD_THRESH

        shadow_mask = (low_brightness & low_saturation & low_contrast)
        return shadow_mask.astype(np.uint8)

    def _cleanup_obstacles(self, obstacle_mask):
        """
        Apply morphological opening and remove small isolated blobs.
        """
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (self.OPEN_KERNEL_SIZE, self.OPEN_KERNEL_SIZE))
        opened = cv2.morphologyEx(obstacle_mask, cv2.MORPH_OPEN, kernel)

        # Connected components filtering by area
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(opened, connectivity=8)
        cleaned = np.zeros_like(opened, dtype=np.uint8)
        h, w = obstacle_mask.shape
        min_area = int(h * w * self.MIN_OBSTACLE_AREA_FRAC)
        for lbl in range(1, num_labels):
            area = stats[lbl, cv2.CC_STAT_AREA]
            if area >= min_area:
                cleaned[labels == lbl] = 1
        return cleaned

    def _perspective_weights(self, h, w):
        """
        Returns a vertical weighting map to encourage forward continuation toward horizon.
        """
        weights = np.ones((h, w), dtype=np.float32)
        upper_third = int(h * 0.33)
        middle = int(h * 0.66)
        for y in range(h):
            if y < upper_third:
                weights[y, :] = self.UPPER_THIRD_WEIGHT
            elif y < middle:
                weights[y, :] = self.MIDDLE_WEIGHT
            else:
                weights[y, :] = self.NEAR_WEIGHT
        return weights

    def _build_cost_map(self, segmentation_mask, image_rgb):
        """
        Convert segmentation to a class-weighted cost map with shadow immunity,
        desert hallucination suppression, and perspective weighting applied.
        Also computes a clearance map (Distance Transform) to encourage centering.
        """
        h, w = segmentation_mask.shape

        # Base class costs
        base_cost = np.zeros((h, w), dtype=np.float32)
        for cls_id, cost in self.CLASS_COST.items():
            base_cost[segmentation_mask == cls_id] = cost

        # Shadow immunity: lower costs in shadow regions for non-hard obstacles
        shadow = self._detect_shadows(image_rgb)
        non_hard = (base_cost < self.CLASS_COST[self.ROCKS_CLASS])  # anything not rocks/logs
        base_cost[shadow == 1] = np.minimum(base_cost[shadow == 1], self.CLASS_COST[1])  # clamp to drivable cost

        # Desert hallucination suppression on hard obstacles
        hard_obs_mask = ((segmentation_mask == self.ROCKS_CLASS) | (segmentation_mask == self.LOGS_CLASS)).astype(np.uint8)
        hard_obs_mask = self._cleanup_obstacles(hard_obs_mask)

        # Update base_cost: keep hard obstacles high; reduce small/noisy regions already cleaned
        base_cost[hard_obs_mask == 1] = np.maximum(base_cost[hard_obs_mask == 1], self.CLASS_COST[self.ROCKS_CLASS])

        # Perspective weighting: reduce penalties in upper third for non-hard obstacles
        persp_w = self._perspective_weights(h, w)
        # Do not reduce costs for hard obstacles
        reduced = (hard_obs_mask == 0)
        cost_map = base_cost.copy()
        cost_map[reduced] = cost_map[reduced] * persp_w[reduced]

        # Horizon guard: mask out sky region above horizon
        horizon_line = int(h * self.HORIZON_RATIO)
        sky_mask = (segmentation_mask == self.SKY_CLASS)
        row_mask = (np.arange(h) < horizon_line)
        cost_map[row_mask, :] = np.inf  # forbid traversal above horizon
        cost_map[sky_mask] = np.inf  # forbid sky anywhere

        # --- Distance Transform for Clearance ---
        # Invert hard obstacle mask (0 for obstacle, 1 for free)
        # We also treat sky/horizon as "obstacles" for clearance purposes to keep path centered
        clearance_mask = (hard_obs_mask == 0).astype(np.uint8)
        clearance_mask[row_mask] = 0
        clearance_mask[sky_mask] = 0
        
        # Compute distance to nearest zero pixel
        dist_map = cv2.distanceTransform(clearance_mask, cv2.DIST_L2, 5)
        
        # Add potential field to cost_map to encourage centering (Corridor-Aware)
        # Cost increases as we get closer to obstacles.
        # k / (dist + epsilon)
        # Stronger weight as requested to prioritize corridor width
        potential_cost = 20.0 / (dist_map + 1.0)
        
        # Only apply potential to traversable areas
        traversable = ~np.isinf(cost_map)
        cost_map[traversable] += potential_cost[traversable]

        return cost_map, hard_obs_mask, shadow, dist_map, base_cost

    def _find_candidate_paths(self, cost_map, seed_point, horizon_row, hard_obs_mask):
        """
        Runs a single Dijkstra flood from seed_point to explore the map.
        Extracts candidate paths by looking for ALL reachable endpoints at the 
        furthest explored frontiers, ensuring equal exploration of Left/Center/Right.
        """
        h, w = cost_map.shape
        inf = np.inf
        start_t = time.perf_counter()
        iterations = 0

        # Initialize
        dist = np.full((h, w), inf, dtype=np.float32)
        prev = np.full((h, w, 2), -1, dtype=np.int32)

        sx, sy = seed_point
        dist[sy, sx] = 0.0
        pq = [(0.0, (sx, sy))]

        # Full 8-connectivity
        moves = [(-1, -1), (0, -1), (1, -1),
                 (-1, 0),           (1, 0),
                 (-1, 1),  (0, 1),  (1, 1)]

        min_y_reached = sy

        while pq:
            # Hard time and iteration caps
            if (time.perf_counter() - start_t) > self.TIME_LIMIT_SEC:
                break
            if iterations > self.MAX_ITERATIONS:
                break

            d, (x, y) = heapq.heappop(pq)
            if d > dist[y, x]:
                continue
            
            if y < min_y_reached:
                min_y_reached = y
            
            iterations += 1

            for dx, dy in moves:
                nx, ny = x + dx, y + dy
                if nx < 0 or nx >= w or ny < 0 or ny >= h:
                    continue
                if np.isinf(cost_map[ny, nx]):
                    continue
                
                step_cost = cost_map[ny, nx]
                # Hard obstacles are strictly untraversable
                if hard_obs_mask[ny, nx] == 1:
                    continue

                nd = d + step_cost
                if nd < dist[ny, nx]:
                    dist[ny, nx] = nd
                    prev[ny, nx] = [x, y]
                    heapq.heappush(pq, (nd, (nx, ny)))

        # Extract Candidates
        candidates = []
        
        # Helper to trace back path
        def trace_path(gx, gy):
            path = []
            cx, cy = gx, gy
            if prev[cy, cx][0] == -1:
                return []
            while True:
                path.append((cx, cy))
                px, py = prev[cy, cx]
                if px == -1:
                    break
                cx, cy = px, py
            path.reverse()
            return path

        # Get all reachable coordinates
        reachable_y, reachable_x = np.where(~np.isinf(dist))
        
        if reachable_y.size == 0:
            return []
            
        # We want to find "local maxima" of reachability in different directions.
        # Strategy: Divide the image into many vertical strips (e.g., 10)
        # In each strip, find the furthest point (min Y).
        # This gives us a set of "furthest reachable points" across the width.
        
        num_strips = 10
        strip_width = w / num_strips
        
        for i in range(num_strips):
            x_start = int(i * strip_width)
            x_end = int((i + 1) * strip_width)
            
            # Mask for this strip
            strip_mask = (reachable_x >= x_start) & (reachable_x < x_end)
            
            if np.any(strip_mask):
                strip_ys = reachable_y[strip_mask]
                strip_xs = reachable_x[strip_mask]
                
                # Find min Y (furthest forward)
                min_y = np.min(strip_ys)
                
                # Get all nodes at min_y in this strip
                target_indices = np.where(strip_ys == min_y)[0]
                target_xs = strip_xs[target_indices]
                
                # Tie-breaker: pick lowest cost
                best_target_idx = np.argmin(dist[min_y, target_xs])
                best_x = target_xs[best_target_idx]
                
                path = trace_path(best_x, min_y)
                if path:
                    candidates.append(path)
        
        return candidates

    def _score_path(self, path, hard_obs_mask, base_cost, dist_map):
        """
        Scores a path using STRICT ORDER TUPLE:
        (mean_width, length, -jerk)
        """
        if not path or len(path) < 2:
            return (-float('inf'), -float('inf'), -float('inf'))

        # 1. Primary: Maximize Corridor Width (Mean Clearance)
        path_arr = np.array(path)
        xs = path_arr[:, 0]
        ys = path_arr[:, 1]
        
        clearance_vals = dist_map[ys, xs]
        mean_lateral_clearance = np.mean(clearance_vals)
        
        # 2. Secondary: Maximize Total Reachable Length
        forward_distance = float(len(path))
        
        # 3. Tertiary: Minimize Curvature Jerk
        curvature_jerk = 0.0
        if len(path) > 2:
            diffs = path_arr[1:] - path_arr[:-1]
            angles = np.arctan2(diffs[:, 1], diffs[:, 0])
            angle_diffs = angles[1:] - angles[:-1]
            angle_diffs = (angle_diffs + np.pi) % (2 * np.pi) - np.pi
            curvature_jerk = np.sum(np.abs(angle_diffs))
            
        # Return tuple for lexicographical comparison
        # (Primary, Secondary, Tertiary)
        # Since we want to minimize jerk, we return -jerk
        return (mean_lateral_clearance, forward_distance, -curvature_jerk)

    def _smooth_path(self, path, h, w):
        """
        Simple polyline smoothing using Blackman window on x coordinates.
        """
        if len(path) <= self.SMOOTHING_WINDOW_SIZE:
            return path
        path_x = [p[0] for p in path]
        path_y = [p[1] for p in path]
        window = np.blackman(self.SMOOTHING_WINDOW_SIZE)
        window = window / np.sum(window)
        pad_len = self.SMOOTHING_WINDOW_SIZE // 2
        path_x_padded = np.pad(path_x, (pad_len, pad_len), mode='edge')
        smooth_x = np.convolve(path_x_padded, window, mode='valid')
        final_path = []
        for i in range(len(smooth_x)):
            sx = int(np.clip(smooth_x[i], 0, w - 1))
            sy = int(np.clip(path_y[i], 0, h - 1))
            final_path.append((sx, sy))
        return final_path

    def _compute_obstacle_density(self, hard_obs_mask, horizon_line):
        """
        Compute obstacle density in forward region after cleanup.
        """
        region = hard_obs_mask[horizon_line:, :]
        total = region.size
        count = int(np.sum(region))
        return (count / total) * 100.0

    def plan_path_costaware(self, segmentation_mask, image_rgb):
        """
        Main entry: builds cost map and plans a cost-aware path toward horizon,
        with shadow immunity, desert noise suppression, perspective weighting,
        path length guarantee, and smoothing. Planning happens at reduced resolution
        and is rescaled back to full resolution for visualization.
        """
        try:
            h, w = segmentation_mask.shape
            hs = max(1, int(h * self.SCALE_FACTOR))
            ws = max(1, int(w * self.SCALE_FACTOR))
            scale = self.SCALE_FACTOR

            # Downscale inputs for fast planning
            seg_small = cv2.resize(segmentation_mask.astype(np.uint8), (ws, hs), interpolation=cv2.INTER_NEAREST)
            img_small = cv2.resize(image_rgb, (ws, hs), interpolation=cv2.INTER_AREA)
            horizon_line_small = int(hs * self.HORIZON_RATIO)

            cost_map, hard_obs_mask, shadow_mask, dist_map, base_cost = self._build_cost_map(seg_small, img_small)

            # Seed Selection: Anchor in WIDEST free-space region at bottom
            # Using Distance Transform map (dist_map)
            # Scan bottom 30% of image to find the "entrance" point with maximum clearance
            scan_rows = int(hs * 0.30)
            scan_y_start = max(0, hs - scan_rows)
            scan_y_end = hs
            
            # ROI of distance map
            roi_dist = dist_map[scan_y_start:scan_y_end, :]
            
            # Mask out non-traversable areas (cost is inf or hard obstacle) in ROI
            roi_cost = cost_map[scan_y_start:scan_y_end, :]
            roi_hard = hard_obs_mask[scan_y_start:scan_y_end, :]
            
            valid_mask = (~np.isinf(roi_cost)) & (roi_hard == 0)
            
            # Apply mask to dist values (set invalid to -1)
            valid_roi_dist = roi_dist.copy()
            valid_roi_dist[~valid_mask] = -1.0
            
            if np.max(valid_roi_dist) < 0:
                 # If bottom is strictly blocked, try scanning the entire lower half
                 # to find *any* start point, even if disconnected from bottom edge.
                 scan_rows = int(hs * 0.60)
                 scan_y_start = max(0, hs - scan_rows)
                 roi_dist = dist_map[scan_y_start:scan_y_end, :]
                 roi_cost = cost_map[scan_y_start:scan_y_end, :]
                 roi_hard = hard_obs_mask[scan_y_start:scan_y_end, :]
                 valid_mask = (~np.isinf(roi_cost)) & (roi_hard == 0)
                 valid_roi_dist = roi_dist.copy()
                 valid_roi_dist[~valid_mask] = -1.0
                 
                 if np.max(valid_roi_dist) < 0:
                     return [], np.zeros((h, w), dtype=np.uint8), hard_obs_mask
                 
            # Find coordinates of max clearance
            # argmax returns linear index
            max_idx = np.unravel_index(np.argmax(valid_roi_dist), valid_roi_dist.shape)
            
            # Convert back to global coordinates
            seed_y = scan_y_start + max_idx[0]
            seed_x = max_idx[1]

            # Obstacle density
            obstacle_density = self._compute_obstacle_density(hard_obs_mask, horizon_line_small)

            # Path length guarantee
            # Removed density cap as requested: always try to reach close to horizon
            goal_row = horizon_line_small

            # Multi-Candidate Search
            candidates = self._find_candidate_paths(cost_map, (seed_x, seed_y), goal_row, hard_obs_mask)
            
            if not candidates:
                # No path found at all (very rare)
                return [], np.zeros((h, w), dtype=np.uint8), hard_obs_mask

            # Evaluate Candidates using REQUIRED FORMULA
            best_path = []
            best_score = (-float('inf'), -float('inf'), -float('inf'))

            for path in candidates:
                score = self._score_path(path, hard_obs_mask, base_cost, dist_map)
                
                if score > best_score:
                    best_score = score
                    best_path = path
            
            path = best_path
            
            # If path is extremely short, it might be effectively blocked,
            # but we still return it as requested ("render longest partial path")
            
            # Rescale path back to full resolution
            path_full = [(int(round(x / scale)), int(round(y / scale))) for (x, y) in path]
            # Clamp to image bounds and smooth
            path_full = [(max(0, min(w - 1, x)), max(0, min(h - 1, y))) for (x, y) in path_full]
            path_full = self._smooth_path(path_full, h, w)

            # Build connectivity mask for visualization (optional)
            path_mask = np.zeros((h, w), dtype=np.uint8)
            for (x, y) in path_full:
                path_mask[y, x] = 1

            return path_full, path_mask, hard_obs_mask

        except Exception as e:
            print(f"Path planning error: {e}")
            import traceback
            traceback.print_exc()
            # Return empty path so visualizer can show preview
            h, w = segmentation_mask.shape
            return [], np.zeros((h, w), dtype=np.uint8), np.zeros((h, w), dtype=np.uint8)

    def visualize_path(self, image, path, color=(0, 255, 255), thickness=3):
        try:
            vis_img = image.copy()
            
            if path and len(path) > 1:
                # VISUALIZATION COORDINATE SYSTEM:
                # Planner generates paths in (x, y) = (col, row) coordinates.
                # OpenCV drawing functions expect (x, y).
                # NO SWAP is needed. Passing (x, y) directly ensures horizontal corridors
                # are drawn horizontally.
                
                pts = np.array(path, np.int32).reshape((-1, 1, 2))
                cv2.polylines(vis_img, [pts], False, color, max(3, thickness))
                cv2.circle(vis_img, path[0], 5, (0, 255, 0), -1)
                cv2.circle(vis_img, path[-1], 5, (0, 0, 255), -1)
            else:
                # Explicit NO SAFE PATH state
                font = cv2.FONT_HERSHEY_SIMPLEX
                text = "NO SAFE PATH"
                font_scale = 1.5
                thickness_text = 4
                text_size = cv2.getTextSize(text, font, font_scale, thickness_text)[0]
                text_x = (vis_img.shape[1] - text_size[0]) // 2
                text_y = (vis_img.shape[0] + text_size[1]) // 2
                cv2.putText(vis_img, text, (text_x, text_y), font, font_scale, (0, 0, 255), thickness_text)
            
            return vis_img
            
        except Exception as e:
            print(f"Visualization error: {e}")
            import traceback
            traceback.print_exc()
            return image # Fail safe


    def clean_mask_for_reasoner(self, segmentation_mask, image_rgb):
        """
        Produce a cleaned mask for status computation:
        - Remove shadow influence for non-hard obstacles
        - Suppress desert noise via morphological opening and area filtering
        """
        h, w = segmentation_mask.shape
        horizon_line = int(h * self.HORIZON_RATIO)

        shadow = self._detect_shadows(image_rgb)
        cleaned = segmentation_mask.copy()
        # In shadow regions, relax obstacles except hard obstacles to Landscape
        relax_mask = (shadow == 1) & (~np.isin(cleaned, list(self.HARD_OBSTACLES)))
        cleaned[relax_mask] = 1

        # Clean hard obstacles
        hard_obs = ((cleaned == self.ROCKS_CLASS) | (cleaned == self.LOGS_CLASS)).astype(np.uint8)
        hard_obs = self._cleanup_obstacles(hard_obs)
        # Update cleaned: keep only cleaned hard obstacles; others to Landscape when tiny
        cleaned[(cleaned == self.ROCKS_CLASS) | (cleaned == self.LOGS_CLASS)] = 1
        cleaned[hard_obs == 1] = self.ROCKS_CLASS  # mark as rocks (represent hard obstacles)

        return cleaned
