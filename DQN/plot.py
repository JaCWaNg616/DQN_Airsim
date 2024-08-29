import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def plot_flight_paths():
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    start_points = []
    end_points = []

    for i in range(1, 11):
        flight_path = np.load(f'C:\\Python project\\DQN\\flight_paths\\flight_path{i}.npy')
        ax.plot(flight_path[:, 0], flight_path[:, 1], flight_path[:, 2])

        # 记录起始点和终点
        start_points.append(flight_path[0])
        end_points.append(flight_path[-1])

    # 计算起始点和终点的平均值
    avg_start_point = np.mean(start_points, axis=0)
    avg_end_point = np.mean(end_points, axis=0)

    # 在图中标出平均起始点和终点，并添加标签
    ax.scatter(avg_start_point[0], avg_start_point[1], avg_start_point[2], color='black', s=10)
    ax.text(avg_start_point[0], avg_start_point[1], avg_start_point[2], 'start', color='black')

    ax.scatter(avg_end_point[0], avg_end_point[1], avg_end_point[2], color='black', s=10)
    ax.text(avg_end_point[0], avg_end_point[1], avg_end_point[2], 'goal', color='black')



    # 设置坐标轴标签
    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')
    ax.set_zlabel('Z Position')

    # 设置标题
    ax.set_title('Drone Trajectories')

    plt.show()


if __name__ == "__main__":
    plot_flight_paths()
