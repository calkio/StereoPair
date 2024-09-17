import numpy as np
import matplotlib.pyplot as plt

def load_coordinates(file_path):
    with open(file_path, 'r') as file:
        data = file.read().replace('[', '').replace(']', '').replace(',', '.')
    points = [list(map(float, p.split(';'))) for p in data.split('\n') if p]
    return np.array(points)

def interpolate_coordinates(coords_0mm, coords_100mm, distance):
    ratio = distance / 100  # Поскольку у нас данные для 0 мм и 100 мм
    interpolated_coords = coords_0mm + ratio * (coords_100mm - coords_0mm)
    return interpolated_coords

def project_point(point_100mm, distance):
    ratio = distance / 100  # Пропорция по расстоянию
    point_projected = point_100mm * ratio
    return point_projected

def visualize_3d_coordinates(coords_0mm, coords_custom, coords_100mm, custom_distance):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Отображаем точки на расстоянии 0 мм
    ax.scatter(coords_0mm[:, 0], coords_0mm[:, 1], 0, c='blue', label='0 mm')

    # Отображаем точки на расстоянии, введённом пользователем
    ax.scatter(coords_custom[:, 0], coords_custom[:, 1], custom_distance, c='green', label=f'{custom_distance} mm')

    # Отображаем точки на расстоянии 100 мм
    ax.scatter(coords_100mm[:, 0], coords_100mm[:, 1], 100, c='red', label='100 mm')

    # Настройки осей
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Distance (mm)')
    ax.set_title('Calibration Grid Nodes at Different Distances')

    ax.legend()
    plt.show()

def visualize_3d_coordinates_with_point(coords_0mm, coords_custom, coords_100mm, custom_distance, point_100mm, point_projected):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Отображаем точки на расстоянии 0 мм
    ax.scatter(coords_0mm[:, 0], coords_0mm[:, 1], 0, c='blue', label='0 mm')

    # Отображаем точки на пользовательском расстоянии
    ax.scatter(coords_custom[:, 0], coords_custom[:, 1], custom_distance, c='green', label=f'{custom_distance} mm')

    # Отображаем точки на расстоянии 100 мм
    ax.scatter(coords_100mm[:, 0], coords_100mm[:, 1], 100, c='red', label='100 mm')

    # Отображаем исходную точку на 100 мм
    ax.scatter(point_100mm[0], point_100mm[1], 100, c='orange', label='Point at 100 mm', s=100, marker='x')

    # Отображаем проецированную точку на пользовательском расстоянии
    ax.scatter(point_projected[0], point_projected[1], custom_distance, c='purple', label=f'Projected Point at {custom_distance} mm', s=100, marker='o')

    # Настройки осей
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Distance (mm)')
    ax.set_title('Calibration Grid Nodes with Projected Point')

    ax.legend()
    plt.show()

def main():
    # Загружаем координаты узлов для 0 мм и 100 мм
    coords_0mm = load_coordinates(r"D:\Dev\Calibrovka\NodesOmsk\0\Nodes.txt")
    coords_100mm = load_coordinates(r"D:\Dev\Calibrovka\NodesOmsk\100\Nodes.txt")

    # Запрашиваем расстояние от пользователя
    distance = float(input("Введите расстояние для генерации калибровочной матрицы (мм): "))

    # Генерируем координаты для указанного расстояния
    interpolated_coords = interpolate_coordinates(coords_0mm, coords_100mm, distance)

    # Запрашиваем координаты точки на расстоянии 100 мм
    point_100mm = np.array([float(input("Введите X координату точки на 100 мм: ")), float(input("Введите Y координату точки на 100 мм: "))])

    # Проецируем точку на пользовательское расстояние
    point_projected = project_point(point_100mm, distance)

    # Визуализируем все точки и проецированную точку
    visualize_3d_coordinates_with_point(coords_0mm, interpolated_coords, coords_100mm, distance, point_100mm, point_projected)
if __name__ == '__main__':
    main()