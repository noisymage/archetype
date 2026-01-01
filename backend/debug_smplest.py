
import sys
import os
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("DebugSMPLest")

# Add backend to sys.path so we can import vision_engine
BACKEND_DIR = Path(__file__).parent.resolve()
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

try:
    from vision_engine import ModelManager, BodyAnalyzer
    print("Successfully imported vision_engine")
except ImportError as e:
    print(f"Failed to import vision_engine: {e}")
    sys.exit(1)

def main():
    print("--- Starting SMPLest-X Debug ---")
    
    manager = ModelManager()
    print(f"Device: {manager.device}")
    
    print("Attempting to load mesh model (this should crash if imports fail)...")
    try:
        success = manager.load_mesh_model()
        if success:
            print("SUCCESS: SMPLest-X model loaded!")
        else:
            print("FAILURE: ModelManager returned False (degraded mode?)")
            
    except Exception as e:
        print(f"CRASHED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
        
    # If successful, run inference
    test_img = BACKEND_DIR / "test_data/body_front.png"
    if not test_img.exists():
        print(f"Test image not found at {test_img}")
        return

    print(f"Running inference on {test_img}...")
    analyzer = BodyAnalyzer(manager)
    # We might need to ensure Pose model is loaded for the bbox detection inside BodyAnalyzer
    manager.load_pose_model()
    
    result = analyzer.analyze(str(test_img))
    
    print("\n--- Analysis Result ---")
    print(f"Analyzed: {result.analyzed}")
    print(f"Degraded: {result.degraded_mode}")
    print(f"Error: {result.error}")
    
    if result.volume_estimate:
        print(f"Volume Estimate: {result.volume_estimate:.2f} Liters")
    else:
        print("Volume Estimate: None")
        
    if result.betas is not None:
        print(f"Betas (first 5): {result.betas[:5]}")

if __name__ == "__main__":
    main()
