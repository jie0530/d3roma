import threading
import time
from utils_d3roma.realsense import RealSenseRGBDCamera

def thread_function(thread_id, camera):
    print(f"Thread {thread_id} started")
    try:
        while True:
            start_time = time.time()
            rgb_frame, depth_aligned = camera.get_rgbd_image()
            end_time = time.time()
            print(f"Thread {thread_id} - Time taken: {end_time - start_time:.4f}s")
            print(f"Thread {thread_id} - RGB shape: {rgb_frame.shape}, Depth shape: {depth_aligned.shape}")
            time.sleep(0.1)  # 控制采样频率
    except Exception as e:
        print(f"Thread {thread_id} error: {e}")

def main():
    # 初始化相机
    camera = RealSenseRGBDCamera(serial="236522072295")
    
    # 预热相机
    print("Warming up camera...")
    for _ in range(30):
        camera.get_rgbd_image()
    print("Camera initialization finished")
    
    # 创建两个线程
    threads = []
    for i in range(2):
        thread = threading.Thread(target=thread_function, args=(i, camera))
        threads.append(thread)
        thread.start()
    
    try:
        # 等待所有线程完成
        for thread in threads:
            thread.join()
    except KeyboardInterrupt:
        print("Stopping threads...")
    finally:
        print("Cleaning up...")

if __name__ == "__main__":
    main()