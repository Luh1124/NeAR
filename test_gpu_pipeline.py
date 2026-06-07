import asyncio
import json
import time
import requests
import websockets
from pathlib import Path

# Config
BASE_URL = "http://127.0.0.1:8099"
WS_URL = "ws://127.0.0.1:8099/ws"
TEST_IMAGE = "/home/hongli/code/3dgen/GeomDist/data/spot/spot_by_keenan.png"

async def test_pipeline():
    print("=" * 60)
    print("NeAR GPU End-to-End Test and Benchmark Tool")
    print("=" * 60)
    
    # 1. Upload image
    print(f"\n[Step 1] Uploading test image: {TEST_IMAGE}...")
    if not Path(TEST_IMAGE).exists():
        print(f"Error: Test image not found at {TEST_IMAGE}")
        return
        
    with open(TEST_IMAGE, "rb") as f:
        files = {"file": f}
        r = requests.post(f"{BASE_URL}/upload", files=files)
        
    if r.status_code != 200:
        print(f"Error: Upload failed with status {r.status_code}: {r.text}")
        return
        
    res = r.json()
    print(f"Upload success! Response: {res}")
    session_id = res["session_id"]
    uploaded_path = res["image_path"]
    
    # 2. Connect to WebSocket
    print(f"\n[Step 2] Connecting to WebSocket: {WS_URL}...")
    async with websockets.connect(WS_URL) as ws:
        # Receive init config
        init_msg_raw = await ws.recv()
        init_msg = json.loads(init_msg_raw)
        print("Received init message from server.")
        print(f"Mock Mode: {init_msg.get('mock_mode')}")
        print(f"Default SLaT: {init_msg.get('current_slat')}")
        print(f"Default HDRI: {init_msg.get('current_hdri')}")
        
        if init_msg.get('mock_mode'):
            print("WARNING: Server is running in MOCK (CPU) MODE! This test aims for GPU verification.")
        else:
            print("SUCCESS: Server is running in REAL GPU MODE!")
            
        # 3. Trigger 3D Generation
        print(f"\n[Step 3] Starting 3D Asset Generation for session {session_id}...")
        gen_payload = {
            "type": "generate",
            "image_path": uploaded_path,
            "seed": 42
        }
        await ws.send(json.dumps(gen_payload))
        
        slat_path = None
        video_url = None
        
        # Loop for progress / completion
        start_time = time.time()
        while True:
            msg_raw = await ws.recv()
            if isinstance(msg_raw, bytes):
                # Ignore random render updates if any
                continue
                
            msg = json.loads(msg_raw)
            msg_type = msg.get("type")
            
            if msg_type == "progress":
                progress = msg.get("progress", 0.0)
                message = msg.get("message", "")
                print(f"  [Progress] {progress*100:.1f}%: {message}")
            elif msg_type == "generation_complete":
                slat_path = msg.get("slat_path")
                video_url = msg.get("video_url")
                print(f"SUCCESS: 3D Asset Generation Completed in {time.time() - start_time:.2f} seconds!")
                print(f"  SLaT path: {slat_path}")
                print(f"  Video URL: {video_url}")
                break
            elif msg_type == "error":
                print(f"ERROR during generation: {msg.get('message')}")
                return
                
        # 4. Perform Interactive Relighting (3DGS Drag Render benchmark)
        print(f"\n[Step 4] Benchmarking Interactive Relighting (3DGS Canvas pixel-stream)...")
        # We will simulate dragging the mouse (yaw changing) for 20 frames
        latencies = []
        for i in range(20):
            test_yaw = (i * 18.0) % 360
            drag_payload = {
                "yaw": test_yaw,
                "pitch": 15.0,
                "radius": 2.0,
                "fov": 40.0,
                "is_dragging": True,
                "hdri_rot_deg": (i * 10.0) % 360,
                "tone_view": "AgX",
                "drag_res": 256,
                "idle_res": 512,
                "slat_path": slat_path,
                "hdri_path": init_msg.get('current_hdri')
            }
            
            frame_start = time.time()
            await ws.send(json.dumps(drag_payload))
            
            # Wait for binary response (image bytes)
            frame_bytes = await ws.recv()
            if not isinstance(frame_bytes, bytes):
                # If we get a text message (e.g. error), parse and show it
                txt_msg = json.loads(frame_bytes)
                print(f"Error frame message: {txt_msg}")
                continue
                
            elapsed = (time.time() - frame_start) * 1000  # ms
            latencies.append(elapsed)
            if i % 5 == 0 or i == 19:
                print(f"  Rendered Frame {i+1}/20 | Yaw: {test_yaw:.1f}° | Size: {len(frame_bytes)/1024:.1f} KB | Latency: {elapsed:.1f}ms")
                
        avg_latency = sum(latencies) / len(latencies)
        print(f"Relighting Benchmark Result:")
        print(f"  Average Frame Rendering + Network roundtrip: {avg_latency:.2f}ms")
        print(f"  Equivalent FPS: {1000.0 / avg_latency:.1f} FPS")
        
        # 5. Export PBR GLB
        print(f"\n[Step 5] Triggering PBR GLB Baking & Export...")
        export_payload = {
            "type": "export_glb",
            "slat_path": slat_path,
            "hdri_path": init_msg.get('current_hdri'),
            "simplify": 0.9,
            "texture_size": 1024
        }
        await ws.send(json.dumps(export_payload))
        
        glb_url = None
        start_time = time.time()
        while True:
            msg_raw = await ws.recv()
            if isinstance(msg_raw, bytes):
                continue
                
            msg = json.loads(msg_raw)
            msg_type = msg.get("type")
            
            if msg_type == "progress":
                progress = msg.get("progress", 0.0)
                message = msg.get("message", "")
                print(f"  [Progress] {progress*100:.1f}%: {message}")
            elif msg_type == "glb_complete":
                glb_url = msg.get("glb_url")
                print(f"SUCCESS: PBR GLB Baked & Exported in {time.time() - start_time:.2f} seconds!")
                print(f"  GLB URL: {glb_url}")
                break
            elif msg_type == "error":
                print(f"ERROR during export: {msg.get('message')}")
                return
                
        # 6. Final verification of files
        print(f"\n[Step 6] Verifying generated assets on disk...")
        # Check files under tmp_web_app/{session_id}
        session_dir = Path(f"tmp_web_app/{session_id}")
        
        # The expected generated files are:
        # - generated_slat.npz
        # - near_pbr_*.glb
        # - camera_path_*.mp4
        slat_file = session_dir / "generated_slat.npz"
        video_files = list(session_dir.glob("camera_path_*.mp4"))
        glb_files = list(session_dir.glob("near_pbr_*.glb"))
        
        if slat_file.exists():
            print(f"  [OK] SLaT .npz exists: {slat_file} ({slat_file.stat().st_size / 1024 / 1024:.2f} MB)")
        else:
            print(f"  [FAIL] SLaT .npz NOT found!")
            
        if video_files:
            video_file = video_files[0]
            print(f"  [OK] Camera video exists: {video_file} ({video_file.stat().st_size / 1024:.2f} KB)")
        else:
            print(f"  [FAIL] Camera video NOT found!")
            
        if glb_files:
            glb_file = glb_files[0]
            print(f"  [OK] PBR GLB exists: {glb_file} ({glb_file.stat().st_size / 1024 / 1024:.2f} MB)")
        else:
            print(f"  [FAIL] PBR GLB NOT found!")
            
    print("\n" + "=" * 60)
    print("ALL GPU TESTS COMPLETED!")
    print("=" * 60)

if __name__ == "__main__":
    asyncio.run(test_pipeline())
