import airsim
import numpy as np
import open3d as o3d
from sklearn.cluster import DBSCAN
from scipy.spatial.transform import Rotation as R
import time
import random
import scipy.stats as stats

COLOR_SEQUENCE = [
    [0.5, 0, 0],  # 深红
    [0, 0.5, 0],  # 深绿
    [0, 0, 0.5],  # 深蓝
    [0.5, 0.5, 0],  # 橄榄
    [0.5, 0, 0.5],  # 紫红
    [0, 0.5, 0.5],  # 深青
    [1, 0.5, 0],  # 橙色
    [0.5, 1, 0],  # 黄绿色
    [0, 0.5, 1],  # 天蓝色
    [1, 0.5, 0.5],  # 浅红
    [0.5, 1, 1],  # 浅青
    [1, 1, 0.5]  # 浅黄
]


def get_lidar_data(client, vehicle_name, lidar_name="LidarSensor1"):
    """
    获取无人机的LiDAR数据
    """
    lidar_data = client.getLidarData(lidar_name=lidar_name, vehicle_name=vehicle_name)
    if len(lidar_data.point_cloud) < 3:
        return np.array([]), np.array([])

    points = np.array(lidar_data.point_cloud, dtype=np.float32).reshape(-1, 3)
    distances = np.linalg.norm(points, axis=1)
    points = points[distances <= 12]  # 筛选出距离在12米以内的点

    # 将右手坐标系（NED）转换为左手坐标系（Open3D）
    points[:, 1] = -points[:, 1]  # Y轴取反
    points[:, 2] = -points[:, 2]  # Z轴取反

    return points, distances


def create_3d_grid(size_x=8, size_y=8, size_z=2, step=2):
    """
    创建3D网格，用于可视化
    """
    points = []

    for x in range(-size_x, size_x + 1, step):
        for y in range(-size_y, size_y + 1, step):
            points.append([x, y, -size_z])
            points.append([x, y, size_z])
    for z in range(-size_z, size_z + 1, step):
        for x in range(-size_x, size_x + 1, step):
            points.append([x, -size_y, z])
            points.append([x, size_y, z])
    for z in range(-size_z, size_z + 1, step):
        for y in range(-size_y, size_y + 1, step):
            points.append([-size_x, y, z])
            points.append([size_x, y, z])

    lines = [[i, i + 1] for i in range(0, len(points), 2)]
    return points, lines


def update_grid_transform(client, line_set, initial_points, vehicle_name="Drone1"):
    """
    根据无人机的姿态和位置更新3D网格的变换，使网格在无人机移动时按反方向移动
    """
    imu_data = client.getImuData(vehicle_name=vehicle_name)
    orientation = imu_data.orientation.to_numpy_array()

    # 四元数转换
    rotation = R.from_quat(orientation).as_matrix()

    # 修正y轴旋转方向
    rotation[1, :] = -rotation[1, :]

    kinematics = client.simGetGroundTruthKinematics(vehicle_name)
    position = kinematics.position.to_numpy_array()

    # 将右手坐标系（NED）转换为左手坐标系（Open3D）
    position[0] = -position[0]  # Y轴取反
    position[2] = -position[2] - 3  # Z轴取反

    rotated_points = np.dot(initial_points, rotation.T)
    transformed_points = rotated_points + position
    line_set.points = o3d.utility.Vector3dVector(transformed_points)


def cluster_points_dbscan(points, eps, min_samples):
    """
    使用DBSCAN算法对点云数据进行聚类
    """
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    labels = dbscan.fit_predict(points)
    return labels


def calculate_cluster_center(points):
    """
    计算聚类中心点
    """
    return np.mean(points, axis=0)


def calculate_cluster_width(points):
    """
    计算聚类点云的宽度（在 x 轴和 y 轴方向的最大最小值之差）
    """
    x_diff = np.max(points[:, 0]) - np.min(points[:, 0])
    y_diff = np.max(points[:, 1]) - np.min(points[:, 1])
    return max(x_diff, y_diff)


def generate_arc_waveform(drone_position, cluster_center, cluster_width, num_points=20):
    """
    生成圆弧波形，点密度靠近cluster_center大，越往两边密度越小
    """
    cluster_width = cluster_width * 0.8
    # 计算起始点到圆心的矢量
    radius_vector = np.array(cluster_center) - np.array(drone_position)
    radius = np.linalg.norm(radius_vector)

    # 计算矢量在x-y平面的投影角度
    base_angle = np.arctan2(radius_vector[1], radius_vector[0])

    # 计算角度范围
    angle_range = np.arctan2(cluster_width, radius)

    # 使用正态分布生成角度
    normal_dist = stats.norm(loc=0, scale=1)  # 正态分布
    cdf_values = np.linspace(0.2, 0.8, num_points)  # 避免极端值
    angles = normal_dist.ppf(cdf_values)  # 获得标准正态分布的分位数
    angles = (angles - np.min(angles)) / (np.max(angles) - np.min(angles))  # 归一化到 [0, 1]
    angles = angle_range * (angles - 0.5) * 2 + base_angle  # 映射到实际角度范围

    # 生成圆弧上的点
    arc_points = []
    for angle in angles:
        x = drone_position[0] + radius * np.cos(angle)
        y = drone_position[1] + radius * np.sin(angle)
        z = cluster_center[2]  # 保持z坐标不变
        arc_points.append([x, y, z])

    return np.array(arc_points)


# 设置初始视点
def set_fixed_view(vis):
    ctr = vis.get_view_control()

    # 计算45° 斜向下的前向矢量并归一化
    front = np.array([0.0, 1.0, 1.0])
    front = front / np.linalg.norm(front)

    ctr.set_front(front.tolist())
    ctr.set_lookat([0.0, 0.0, 0.0])
    ctr.set_up([0.0, 0.0, 1.0])  # 计算相应的上向矢量
    ctr.set_zoom(0.8)


def update_point_cloud(client, vis, pcd, line_set, initial_grid_points):
    points, distances = get_lidar_data(client, "Drone1")

    if points.size > 0:
        distances = np.linalg.norm(points, axis=1)
        normal_vector = get_drone_orientation(client, "Drone1")
        projection_distances = np.dot(points, normal_vector)
        min_projection = np.min(projection_distances)
        floor_threshold = min_projection + 0.5
        floor_points = projection_distances <= floor_threshold
        all_labels = np.full(points.shape[0], -1, dtype=int)
        all_labels[floor_points] = -2
        remaining_points = points[~floor_points]
        remaining_distances = distances[~floor_points]
        colors = np.zeros((points.shape[0], 3))
        colors[floor_points] = [0.5, 0.5, 0.5]
        masks = [
            remaining_distances <= 4,
            (remaining_distances > 4) & (remaining_distances <= 6),
            (remaining_distances > 6) & (remaining_distances <= 8)
        ]
        params = [
            (0.3, 25),
            (0.35, 25),
            (0.5, 25)
        ]

        cluster_centers = []
        cluster_widths = []

        for mask, (eps, min_samples) in zip(masks, params):
            if np.any(mask):
                labels = cluster_points_dbscan(remaining_points[mask], eps, min_samples)
                unique_labels = set(labels)
                for label in unique_labels:
                    if label != -1:
                        obstacle_indices = np.where(mask)[0][labels == label]
                        original_indices = np.where(~floor_points)[0][obstacle_indices]
                        all_labels[original_indices] = label
                        cluster_center = calculate_cluster_center(remaining_points[mask][labels == label])
                        cluster_width = calculate_cluster_width(remaining_points[mask][labels == label])
                        cluster_centers.append(cluster_center)
                        cluster_widths.append(cluster_width)

        for label in set(all_labels):
            if label >= 0:
                obstacle_points = (all_labels == label)
                color_index = label % len(COLOR_SEQUENCE)
                colors[obstacle_points] = COLOR_SEQUENCE[color_index]

        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors)
        vis.update_geometry(pcd)

        kinematics = client.simGetGroundTruthKinematics("Drone1")
        drone_position = kinematics.position.to_numpy_array()

        set_fixed_view(vis)
        vis.clear_geometries()
        vis.add_geometry(pcd)
        vis.add_geometry(line_set)

        # 添加基本的圆弧点并设置颜色为蓝色
        for center, width in zip(cluster_centers, cluster_widths):
            arc_points = generate_arc_waveform(drone_position, center, width)
            arc_pcd = o3d.geometry.PointCloud()
            arc_pcd.points = o3d.utility.Vector3dVector(arc_points)
            arc_colors = np.array([[0.6, 0.8, 1.0] for _ in range(len(arc_points))])  # 设置颜色为蓝色
            arc_pcd.colors = o3d.utility.Vector3dVector(arc_colors)
            vis.add_geometry(arc_pcd)

    update_grid_transform(client, line_set, initial_grid_points)
    vis.update_geometry(line_set)
    set_fixed_view(vis)
    vis.poll_events()
    vis.update_renderer()


def get_drone_orientation(client, vehicle_name="Drone1"):
    """
    获取无人机的姿态并计算法向量
    """
    imu_data = client.getImuData(vehicle_name=vehicle_name)
    orientation = imu_data.orientation.to_numpy_array()
    rotation_matrix = R.from_quat(orientation).as_matrix()
    # 法向量是 z 轴方向
    normal_vector = rotation_matrix[:, 2]
    # normal_vector[0] = -normal_vector[0]
    normal_vector[1] = -normal_vector[1]
    # 将 normal_vector 的每个数据缩小到两位小数
    normal_vector = np.round(normal_vector, 4)
    return normal_vector


def move_drone_randomly(client):
    """
    随机移动无人机
    """
    vx = random.uniform(-1, 1)
    vy = random.uniform(-1, 1)
    z = -2
    duration = 1
    client.moveByVelocityZAsync(vx, vy, z, duration, drivetrain=airsim.DrivetrainType.MaxDegreeOfFreedom,
                                yaw_mode=airsim.YawMode(False, 0))


# 创建红色长方体
def create_red_box():
    box = o3d.geometry.TriangleMesh.create_box(width=0.4, height=0.4, depth=0.1)
    box.paint_uniform_color([1, 0, 0])  # 红色
    box.translate([-0.2, -0.05, -0.2])  # 将长方体的中心移动到原点
    return box


def main():
    """
    主函数，设置无人机，初始化可视化，并循环更新点云和网格
    """
    client = airsim.MultirotorClient()
    client.confirmConnection()
    client.enableApiControl(True, "Drone1")
    client.armDisarm(True, "Drone1")

    client.takeoffAsync(vehicle_name="Drone1").join()
    client.moveToZAsync(-2, 1, vehicle_name="Drone1").join()

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name='Point Cloud and 3D Grid', width=800, height=600)
    pcd = o3d.geometry.PointCloud()
    vis.add_geometry(pcd)

    points, lines = create_3d_grid()
    initial_grid_points = np.array(points)
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(points)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    vis.add_geometry(line_set)

    # 创建红色长方体并添加到可视化窗口
    red_box = create_red_box()
    vis.add_geometry(red_box)

    points, distances = get_lidar_data(client, "Drone1")
    if points.size > 0:
        pcd.points = o3d.utility.Vector3dVector(points)
        vis.add_geometry(pcd)
        vis.poll_events()
        vis.update_renderer()

    try:
        while True:
            update_point_cloud(client, vis, pcd, line_set, initial_grid_points)
            move_drone_randomly(client)
            for _ in range(10):
                update_point_cloud(client, vis, pcd, line_set, initial_grid_points)
                time.sleep(0.1)

    except KeyboardInterrupt:
        pass

    finally:
        vis.destroy_window()
        client.landAsync(vehicle_name="Drone1").join()
        client.armDisarm(False, "Drone1")
        client.enableApiControl(False, "Drone1")


if __name__ == "__main__":
    main()
