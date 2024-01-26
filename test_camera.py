import multiprocessing
import threading
import time
import cv2

count_lock = threading.Lock()
shared_count = multiprocessing.Value('i', 0)
shared_stop_camera = multiprocessing.Value('b', False)


def show_camera(count, stop_camera):
    print("LOG: show_camera_thread start")
    # 打开摄像头（默认摄像头，如果有多个摄像头可以指定编号，例如0、1、2等）
    cap = cv2.VideoCapture(0)
    print("LOG: show_camera_thread start2")
    while True:
        # 读取摄像头画面帧
        ret, frame = cap.read()

        # 显示画面
        cv2.imshow('Camera', frame)
        if count.value >= 10:
            cv2.imwrite('file/screenshot10.jpg', frame)
            count.value = 0

        # 检测按键，如果按下 's' 键则保存当前帧为图片
        if cv2.waitKey(1) & 0xFF == ord('s'):
            # 保存当前帧为图片
            cv2.imwrite('file/screenshot.jpg', frame)
            print("截图已保存为 file/screenshot.jpg")

        # 检测按键，如果按下 'q' 键则退出循环
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 释放摄像头资源
    cap.release()

    # 关闭所有OpenCV窗口
    cv2.destroyAllWindows()

    stop_camera.value = True
    print(f"LOG: show_camera_thread end. {stop_camera.value}")


# 线程函数，用于增加计数器的值
def increment_count(count, stop_camera):
    while not stop_camera.value:
        time.sleep(1)
        count.value += 1
        print(f"increment_count {stop_camera.value} {count.value}")


def main2():
    show_camera_process = multiprocessing.Process(
        target=show_camera, args=(shared_count, shared_stop_camera))
    show_camera_process.daemon = True
    show_camera_process.start()
    count_process = multiprocessing.Process(
        target=increment_count, args=(shared_count, shared_stop_camera))
    count_process.daemon = True
    count_process.start()
    show_camera_process.join()
    count_process.join()
    print("LOG: 程序执行结束。")


if __name__ == "__main__":
    main2()
