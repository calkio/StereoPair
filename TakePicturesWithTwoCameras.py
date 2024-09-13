import cv2
import numpy as np
import tkinter as tk
from tkinter import messagebox
import os
from datetime import datetime


# Функция для настройки камеры на максимальное разрешение и установку фокуса на 0
def configure_camera(cap):
    # Получение максимального разрешения (может потребоваться тестирование для конкретной камеры)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 3264)  # Ширина
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 2448)  # Высота
    cap.set(cv2.CAP_PROP_CONTRAST, 50)  # Контраст
    cap.set(cv2.CAP_PROP_BRIGHTNESS, -36)  # Яркость
    cap.set(cv2.CAP_PROP_SATURATION, 52)  # Насыщенность
    cap.set(cv2.CAP_PROP_EXPOSURE, -7)  # Экспозиция
    # Установка фокуса на 0 (если поддерживается)
    if cap.set(cv2.CAP_PROP_FOCUS, 0):
        print("focus set to 0.")
    else:
        print("focus not supported.")

def adjust_brightness_contrast(image, brightness=-18, contrast=50):
    # Применение преобразования яркости и контраста
    new_image = np.clip((1 + contrast/127.0) * image - contrast + brightness, 0, 255).astype(np.uint8)
    return new_image

# Функция для получения изображения с камер и сохранения их
def take_photos():
    # Открываем первую камеру (камера 0)
    cap1 = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap1.isOpened():
        print("cam 1 not opened")
    else:
        print("cam 1 opened")
    
    # Открываем вторую камеру (камера 1)
    cap2 = cv2.VideoCapture(1, cv2.CAP_DSHOW)
    if not cap2.isOpened():
        print("cam 2 not opened")
    else:
        print("cam 2 opened")

    # Настраиваем камеры на максимальное разрешение и устанавливаем фокус на 0
    configure_camera(cap1)
    configure_camera(cap2)

    # Читаем изображение с первой камеры
    ret1, frame1 = cap1.read()
    if not ret1:
        messagebox.showerror("Ошибка", "Не удалось получить изображение с первой камеры")
        cap1.release()
        cap2.release()
        return

    # Читаем изображение со второй камеры
    ret2, frame2 = cap2.read()
    if not ret2:
        messagebox.showerror("Ошибка", "Не удалось получить изображение со второй камеры")
        cap1.release()
        cap2.release()
        return

    # Закрываем камеры
    cap1.release()
    cap2.release()

    # Генерируем имена файлов с текущей датой и временем
    time_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path1 = os.path.join("D:\Dev\StereoPair\img\sterio\left_cam", f"cam1_{time_stamp}.jpg")
    path2 = os.path.join("D:\Dev\StereoPair\img\sterio\\right_cam", f"cam2_{time_stamp}.jpg")

    # Сохраняем изображения
    cv2.imwrite(path1, frame1)
    cv2.imwrite(path2, frame2)

    # Уведомляем пользователя, что изображения сохранены
    # messagebox.showinfo("Успех", f"Фотографии сохранены:\n{path1}\n{path2}")



# Создаем основное окно приложения
root = tk.Tk()
root.title("Фотографирование с двух камер")

# Создаем кнопку для фотографирования
btn = tk.Button(root, text="Сделать фото", command=take_photos)
btn.pack(pady=20)

# Запускаем основное окно
root.mainloop()
print('capture done')

# def list_cameras():
#     index = 0
#     arr = []
#     while True:
#         cap = cv2.VideoCapture(index)
#         if not cap.isOpened():
#             break
#         arr.append(index)
#         cap.release()
#         index += 1
#     return arr

# cameras = list_cameras()
# print(f"Доступные камеры: {cameras}")



# import cv2
# import numpy as np
# import tkinter as tk
# from tkinter import messagebox
# import os
# from datetime import datetime
# import threading

# # Функция для настройки камеры
# def configure_camera(cap, cam_num, resolution=(3264, 2448)):
#     cap.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[0])
#     cap.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])
#     if cap.set(cv2.CAP_PROP_FOCUS, 0):
#         print(f"Camera {cam_num}: focus set to 0.")
#     else:
#         print(f"Camera {cam_num}: focus not supported.")
    
#     # Проверка текущих настроек
#     width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
#     height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
#     focus = cap.get(cv2.CAP_PROP_FOCUS)
#     print(f"Camera {cam_num} settings -> Width: {width}, Height: {height}, Focus: {focus}")

# # Функция для сохранения изображений
# def save_photos():
#     # Настраиваем камеры на максимальное разрешение перед сохранением кадров
#     configure_camera(cap1, 1, (3264, 2448))
#     configure_camera(cap2, 2, (3264, 2448))

#     # Чтение кадров
#     ret1, frame1 = cap1.read()
#     ret2, frame2 = cap2.read()

#     if ret1 and ret2:
#         time_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#         path1 = os.path.join("D:\\Dev\\StereoPair\\img\\sterio\\left_cam", f"cam1_{time_stamp}.jpg")
#         path2 = os.path.join("D:\\Dev\\StereoPair\\img\\sterio\\right_cam", f"cam2_{time_stamp}.jpg")
        
#         cv2.imwrite(path1, frame1)
#         cv2.imwrite(path2, frame2)
        
#         messagebox.showinfo("Успех", f"Фотографии сохранены:\n{path1}\n{path2}")

# # Функция для отображения кадров с двух камер в реальном времени
# def show_stream():
#     # Используем низкое разрешение для стрима
#     configure_camera(cap1, 1, (640, 480))
#     configure_camera(cap2, 2, (640, 480))

#     while True:
#         ret1, frame1 = cap1.read()
#         # ret2, frame2 = cap2.read()

#         if ret1:
#             # Отображаем кадры с двух камер
#             imS = cv2.resize(frame1, (960, 540))
#             cv2.imshow('Left Camera', imS)
#             # cv2.imshow('Right Camera', frame2)

#             # Проверяем, если была нажата клавиша "q" для выхода
#             if cv2.waitKey(1) & 0xFF == ord('q'):
#                 break

#     cap1.release()
#     cap2.release()
#     cv2.destroyAllWindows()

# # Функция для запуска стрима в отдельном потоке
# def start_stream_thread():
#     stream_thread = threading.Thread(target=show_stream)
#     stream_thread.daemon = True
#     stream_thread.start()

# # Открываем камеры
# cap1 = cv2.VideoCapture(0, cv2.CAP_DSHOW)
# cap2 = cv2.VideoCapture(1, cv2.CAP_DSHOW)

# # Проверяем, открылись ли камеры
# if not cap1.isOpened():
#     messagebox.showerror("Ошибка", "Не удалось открыть первую камеру")
#     exit()

# if not cap2.isOpened():
#     messagebox.showerror("Ошибка", "Не удалось открыть вторую камеру")
#     exit()

# # Создаем основное окно приложения
# root = tk.Tk()
# root.title("Стрим с двух камер")

# # Создаем кнопку для сохранения фото
# btn_save = tk.Button(root, text="Сохранить фото", command=save_photos)
# btn_save.pack(pady=20)

# # Запускаем показ стрима в отдельном потоке
# start_stream_thread()

# # Запускаем главное окно
# root.mainloop()

# # Освобождаем ресурсы после завершения работы
# cap1.release()
# cap2.release()
# cv2.destroyAllWindows()
