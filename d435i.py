# Intel RealSense D435i 测试脚本 - 保存到D盘指定文件夹
import pyrealsense2 as rs
import numpy as np
import cv2
import time
import os


def print_h1(name):
    """
    打印标题信息blac
    """
    print("=" * 50)
    print(f"=== {name} ===")
    print("=" * 50)


def detect_realsense_devices():
    """
    检测连接的 RealSense 设备
    """
    print_h1("设备检测")

    detected_devices = []  # 重命名以避免隐藏外部变量

    try:
        # 创建上下文对象
        ctx = rs.context()
        devices = ctx.query_devices()

        print(f"找到 {len(devices)} 个 RealSense 设备")

        # 列出所有设备信息
        for i, dev in enumerate(devices):
            print(f"\n设备 {i}:")
            print(f"  名称: {dev.get_info(rs.camera_info.name)}")
            print(f"  序列号: {dev.get_info(rs.camera_info.serial_number)}")
            print(f"  固件版本: {dev.get_info(rs.camera_info.firmware_version)}")
            print(f"  物理端口: {dev.get_info(rs.camera_info.physical_port)}")
            detected_devices.append(dev)

        return detected_devices

    except Exception as e:
        print(f"设备检测错误: {e}")
        return detected_devices


def get_object_name():
    """
    获取用户输入的物品名称
    """
    print_h1("设置物品名称")
    object_name = input("请输入物品名称（例如：apple, cup, book）: ").strip()
    while not object_name:
        print("物品名称不能为空，请重新输入!")
        object_name = input("请输入物品名称: ").strip()

    print(f"已设置物品名称: {object_name}")
    return object_name


def get_save_directory():
    """
    获取保存目录，默认为D盘的指定文件夹
    """
    # 默认保存到D盘的Data文件夹
    default_dir = "D:/Data"

    # 检查D盘是否存在
    if not os.path.exists("D:/"):
        print("警告: D盘不存在，将保存到当前目录")
        default_dir = "Data"

    # 创建目录（如果不存在）
    os.makedirs(default_dir, exist_ok=True)

    print(f"图像将保存到: {os.path.abspath(default_dir)}")
    return default_dir


def test_camera_functionality(object_name, base_save_dir):
    """
    测试相机功能：捕获并显示RGB和深度图像
    """
    print_h1("相机功能测试")

    # 创建保存目录 - 使用D盘指定文件夹
    save_dir = os.path.join(base_save_dir, object_name)
    os.makedirs(save_dir, exist_ok=True)
    print(f"图像将保存到: {os.path.abspath(save_dir)}")

    # 创建管道和配置
    pipeline = rs.pipeline()
    config = rs.config()

    # 初始化帧计数器
    frame_counter = 0
    save_count = 0

    try:
        # 启用彩色和深度流
        config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 15)
        config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 15)
        # 启动管道
        pipeline_profile = pipeline.start(config)
        device = pipeline_profile.get_device()

        print("相机启动成功!")
        print(f"使用设备: {device.get_info(rs.camera_info.name)}")

        # 创建对齐对象（将深度帧对齐到彩色帧）
        align_to = rs.stream.color
        align = rs.align(align_to)

        # 创建颜色映射对象（用于深度可视化）
        colorizer = rs.colorizer()

        # 创建窗口
        window_name_rgb = f'RealSense D435i - {object_name} - RGB'
        window_name_depth = f'RealSense D435i - {object_name} - Depth'

        cv2.namedWindow(window_name_rgb, cv2.WINDOW_NORMAL)
        cv2.namedWindow(window_name_depth, cv2.WINDOW_NORMAL)

        # 调整窗口大小
        cv2.resizeWindow(window_name_rgb, 640, 480)
        cv2.resizeWindow(window_name_depth, 640, 480)

        # 移动窗口位置，确保它们可见
        cv2.moveWindow(window_name_rgb, 100, 100)
        cv2.moveWindow(window_name_depth, 800, 100)

        print("\n重要提示: 请点击 OpenCV 窗口使其获得焦点!")
        print("按 's' 键保存当前帧")
        print("按 'c' 键连续保存（每帧都保存）")
        print("按 'q' 键退出")
        print("请不要按 Ctrl+C，使用 'q' 键退出程序")

        frame_count = 0
        continuous_save = False
        start_time = time.time()
        last_key_time = time.time()

        while True:
            # 获取帧数据 - 添加超时处理
            try:
                frames = pipeline.wait_for_frames()  # 100ms超时
            except RuntimeError as e:
                print(f"获取帧超时: {e}")
                continue

            # 将对齐深度帧与彩色帧对齐
            try:
                aligned_frames = align.process(frames)
                aligned_depth_frame = aligned_frames.get_depth_frame()
                color_frame = aligned_frames.get_color_frame()
            except RuntimeError as e:
                print(f"帧对齐错误: {e}")
                continue

            if not aligned_depth_frame or not color_frame:
                print("获取到的帧不完整，跳过...")
                continue

            # 转换图像格式
            depth_image = np.asanyarray(aligned_depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())
            depth_colormap = np.asanyarray(colorizer.colorize(aligned_depth_frame).get_data())

            # 显示图像
            cv2.imshow(window_name_rgb, color_image)
            cv2.imshow(window_name_depth, depth_colormap)

            # 连续保存模式
            if continuous_save:
                frame_counter += 1
                rgb_filename = os.path.join(save_dir, f"{object_name}_1_{frame_counter}_rgb.png")
                depth_filename = os.path.join(save_dir, f"{object_name}_1_{frame_counter}_depth.png")
                depth_raw_filename = os.path.join(save_dir, f"{object_name}_1_{frame_counter}_depth_raw.npy")

                # 打印调试信息
                print(f"尝试保存到: {rgb_filename}")

                rgb_success = cv2.imwrite(rgb_filename, color_image)
                depth_success = cv2.imwrite(depth_filename, depth_colormap)
                if rgb_success and depth_success:
                    np.save(depth_raw_filename, depth_image.astype(np.uint16))
                    save_count += 1
                    print(f"已连续保存 {save_count} 帧: {object_name}_{frame_counter}")
                else:
                    if not rgb_success:
                        print(f"错误: 无法保存RGB图像 {rgb_filename}")
                        # 检查目录是否存在
                        if not os.path.exists(save_dir):
                            print(f"目录不存在，尝试创建: {save_dir}")
                            os.makedirs(save_dir, exist_ok=True)
                            # 重试保存
                            rgb_success = cv2.imwrite(rgb_filename, color_image)
                            if rgb_success:
                                print(f"重试成功: 已保存RGB图像 {rgb_filename}")
                    if not depth_success:
                        print(f"错误: 无法保存深度图像 {depth_filename}")

            # 计算并显示帧率
            frame_count += 1
            if frame_count % 30 == 0:
                elapsed_time = time.time() - start_time
                fps = 30 / elapsed_time
                print(f"帧率: {fps:.2f} FPS")
                start_time = time.time()
                frame_count = 0

            # 处理键盘输入 - 使用更短的等待时间
            current_time = time.time()
            if current_time - last_key_time > 0.05:  # 每50ms检查一次键盘
                key = cv2.waitKey(1) & 0xFF
                last_key_time = current_time

                # 调试信息：显示按下的键
                if key != 255:
                    print(f"按键检测: {chr(key) if 32 <= key <= 126 else f'ASCII: {key}'}")

                # 处理按键
                if key == ord('q'):  # 'q' 键退出
                    print("正在退出程序...")
                    break
                elif key == ord('s'):  # 保存当前帧
                    frame_counter += 1
                    rgb_filename = os.path.join(save_dir, f"{object_name}_{frame_counter}_rgb.png")
                    depth_filename = os.path.join(save_dir, f"{object_name}_{frame_counter}_depth.png")
                    depth_raw_filename = os.path.join(save_dir, f"{object_name}_{frame_counter}_depth_raw.npy")

                    # 打印调试信息
                    print(f"尝试保存到: {rgb_filename}")

                    rgb_success = cv2.imwrite(rgb_filename, color_image)
                    depth_success = cv2.imwrite(depth_filename, depth_colormap)
                    if rgb_success and depth_success:
                        np.save(depth_raw_filename, depth_image.astype(np.uint16))
                        save_count += 1
                        print(f"已保存第 {save_count} 帧: {object_name}_{frame_counter}")
                    else:
                        if not rgb_success:
                            print(f"错误: 无法保存RGB图像 {rgb_filename}")
                            # 检查目录是否存在
                            if not os.path.exists(save_dir):
                                print(f"目录不存在，尝试创建: {save_dir}")
                                os.makedirs(save_dir, exist_ok=True)
                                # 重试保存
                                rgb_success = cv2.imwrite(rgb_filename, color_image)
                                if rgb_success:
                                    print(f"重试成功: 已保存RGB图像 {rgb_filename}")
                        if not depth_success:
                            print(f"错误: 无法保存深度图像 {depth_filename}")
                elif key == ord('c'):  # 切换连续保存模式
                    continuous_save = not continuous_save
                    mode = "开启" if continuous_save else "关闭"
                    print(f"连续保存模式已{mode}")

    except Exception as e:
        print(f"相机测试失败: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("正在释放资源...")
        pipeline.stop()
        cv2.destroyAllWindows()
        print(f"相机已停止，共保存了 {save_count} 帧 {object_name} 图像")


# 主程序
if __name__ == '__main__':
    print_h1("Intel RealSense D435i 测试程序 - 保存到D盘")

    # 获取保存目录
    base_save_dir = get_save_directory()

    # 检测设备
    devices_list = detect_realsense_devices()

    if len(devices_list) > 0:
        # 获取物品名称
        object_name = get_object_name()

        # 测试相机功能
        test_camera_functionality(object_name, base_save_dir)
    else:
        print("未检测到 RealSense 设备，请检查连接")

    print_h1("测试完成")