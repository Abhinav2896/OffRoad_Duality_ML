import numpy as np

class NLPReasoner:
    def __init__(self):
        # Hazard Classes
        self.HAZARD_CLASSES = [4, 5, 6, 7] # Rocks, Logs, Bushes, Trees
        # Class 4: Rocks
        # Class 5: Logs
        # Class 6: Bushes
        # Class 7: Trees
        
        self.CLASS_NAMES = {
            4: "Rocks",
            5: "Logs",
            6: "Bushes",
            7: "Trees"
        }
        
    def analyze_scene(self, segmentation_mask):
        """
        Input: segmentation_mask (H, W) with class IDs 0-7
        Output: 
            status: READY / CAUTION / BLOCKED
            report: String description
            metrics: Dict of percentages
        """
        total_pixels = segmentation_mask.size
        
        hazard_pixels = 0
        metrics = {}
        
        for cls_id in self.HAZARD_CLASSES:
            count = np.sum(segmentation_mask == cls_id)
            pct = (count / total_pixels) * 100
            metrics[self.CLASS_NAMES[cls_id]] = pct
            hazard_pixels += count
            
        total_hazard_pct = (hazard_pixels / total_pixels) * 100
        metrics["Total Hazards"] = total_hazard_pct
        
        # Decision Logic
        if total_hazard_pct < 15.0:
            status = "READY"
            report = f"READY: Clear terrain detected ({total_hazard_pct:.1f}% hazards). Path is safe."
        elif 15.0 <= total_hazard_pct <= 50.0:
            status = "CAUTION"
            # Identify dominant hazard
            dominant = max(self.HAZARD_CLASSES, key=lambda c: metrics[self.CLASS_NAMES[c]])
            dom_name = self.CLASS_NAMES[dominant]
            report = f"CAUTION: Moderate hazard density ({total_hazard_pct:.1f}%). Watch for {dom_name}."
        else:
            status = "BLOCKED"
            report = f"BLOCKED: High obstacle density ({total_hazard_pct:.1f}%). Path unsafe."
            
        return {
            "status": status,
            "report": report,
            "metrics": metrics
        }
