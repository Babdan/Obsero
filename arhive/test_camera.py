import cv2
import time

def run_camera_diagnostics():
    """
    Systematically tests camera indices and backends to find a working configuration.
    """
    print("--- Camera Diagnostics ---")
    print("This script will test different settings to find a stable camera feed.")
    print("Please ensure no other applications (Zoom, OBS, etc.) are using the camera.")
    
    backends_to_try = {
        "DSHOW": cv2.CAP_DSHOW,
        "MSMF": cv2.CAP_MSMF,
    }
    
    working_configs = []

    for backend_name, backend_id in backends_to_try.items():
        for index in range(3):
            print(f"\nTesting Index: {index}, Backend: {backend_name}...")
            
            cap = cv2.VideoCapture(index, backend_id)
            
            if not cap.isOpened():
                print("  -> Failed to open.")
                continue

            # Try to set a common resolution and FPS
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            cap.set(cv2.CAP_PROP_FPS, 30)

            # Check if we can read a frame
            ret, frame = cap.read()
            if ret and frame is not None:
                h, w, _ = frame.shape
                print(f"  -> ✓ SUCCESS! Opened with resolution {w}x{h}.")
                working_configs.append({
                    "index": index,
                    "backend_name": backend_name,
                    "backend_id": backend_id,
                    "width": w,
                    "height": h,
                    "cap": cap
                })
            else:
                print("  -> ✗ Opened, but could not read a frame.")
                cap.release()

    if not working_configs:
        print("\n--- Results ---")
        print("❌ No working camera configuration found.")
        print("Suggestions:")
        print("  1. Try a different USB port (preferably a USB 3.0 port).")
        print("  2. Update your camera drivers via Device Manager.")
        print("  3. Ensure your camera is not in use by another program.")
        return

    print("\n--- Found Working Configurations ---")
    for i, config in enumerate(working_configs):
        print(f"  {i+1}. Index: {config['index']}, Backend: {config['backend_name']}, Resolution: {config['width']}x{config['height']}")

    # Let user choose which config to test
    choice = 0
    if len(working_configs) > 1:
        try:
            choice = int(input(f"\nEnter the number of the configuration to test (1-{len(working_configs)}): ")) - 1
            if not 0 <= choice < len(working_configs):
                choice = 0
        except:
            choice = 0
    
    selected_config = working_configs[choice]
    print(f"\nTesting live feed for: Index {selected_config['index']}, Backend {selected_config['backend_name']}...")
    
    cap = selected_config["cap"]
    window_name = f"Live Test: Index {selected_config['index']} ({selected_config['backend_name']})"
    
    frame_count = 0
    start_time = time.time()
    fps = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Lost connection to camera.")
            break

        frame_count += 1
        elapsed = time.time() - start_time
        if elapsed >= 1.0:
            fps = frame_count / elapsed
            frame_count = 0
            start_time = time.time()

        fps_text = f"FPS: {fps:.1f}"
        cv2.putText(frame, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        cv2.imshow(window_name, frame)

        if cv2.waitKey(1) & 0xFF in (ord('q'), ord('Q'), 27):
            break
            
    print("Closing test window.")
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_camera_diagnostics()
