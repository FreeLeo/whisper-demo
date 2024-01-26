import multiprocessing
import time

# 创建共享数据：一个整数和一个数组
shared_number = multiprocessing.Value('d', 0.0)  # 双精度浮点数
shared_array = multiprocessing.Array('i', range(5))  # 整数数组


def modify_data():
    """ 修改共享数据的进程函数 """
    shared_number.value = 3.1415
    for i in range(len(shared_array)):
        shared_array[i] = -shared_array[i]
    print("Data modified in modify_data process.")


def print_data(n, arr):
    """ 打印共享数据的进程函数 """
    print(f"Value: {n.value}")
    print(f"Array: {arr[:]}")


if __name__ == '__main__':
    # 创建并启动修改数据的进程
    p1 = multiprocessing.Process(
        target=modify_data)
    p1.start()
    p1.join()  # 等待p1进程完成

    # 创建并启动打印数据的进程
    p2 = multiprocessing.Process(
        target=print_data, args=(shared_number, shared_array))
    p2.start()
    p2.join()  # 等待p2进程完成
