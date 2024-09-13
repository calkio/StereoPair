import cv2
import tkinter as tk
from tkinter import messagebox
import os
from datetime import datetime
import threading

def take_photos():
    ret1, frame1 = cap1.read()
    if not ret1:
        messagebox.showerror("Ошибка", "Не удалось получить изображение с первой камеры")
        cap1.release()
        cap2.release()
        return

    ret2, frame2 = cap2.read()
    if not ret2:
        messagebox.showerror("Ошибка", "Не удалось получить изображение со второй камеры")
        cap1.release()
        cap2.release()
        return

    time_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path1 = os.path.join(r"D:\Dev\StereoPair\img\sterio\left_cam", f"cam1_{time_stamp}.jpg")
    path2 = os.path.join(r"D:\Dev\StereoPair\img\sterio\right_cam", f"cam2_{time_stamp}.jpg")
    cv2.imwrite(path1, frame1)
    cv2.imwrite(path2, frame2)

def show_stream():
    while True:
        ret, img1 = cap1.read()
        ret, img2 = cap2.read()
        scale_factor = 0.25
        img_resized1 = cv2.resize(img1, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_AREA)
        img_resized2 = cv2.resize(img2, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_AREA)
        cv2.imshow("camera1", img_resized1)
        cv2.imshow("camera2", img_resized2)

        if cv2.waitKey(10) == 27:
            break


def start_stream_thread():
    stream_thread = threading.Thread(target=show_stream)
    stream_thread.daemon = True
    stream_thread.start()

cap1 = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap1.set(cv2.CAP_PROP_FRAME_WIDTH, 3264)  # Ширина
cap1.set(cv2.CAP_PROP_FRAME_HEIGHT, 2448)  # Высота
cap1.set(cv2.CAP_PROP_FOCUS, 0)

cap2 = cv2.VideoCapture(1, cv2.CAP_DSHOW)
cap2.set(cv2.CAP_PROP_FRAME_WIDTH, 3264)  # Ширина
cap2.set(cv2.CAP_PROP_FRAME_HEIGHT, 2448)  # Высота
cap2.set(cv2.CAP_PROP_FOCUS, 0)

# Создаем основное окно приложения
root = tk.Tk()
root.geometry("200x200")
root.title("Фотографирование с двух камер")

btn = tk.Button(root, text="Сделать фото", command=take_photos)
btn.pack()

start_stream_thread()

# Запускаем главное окно
root.mainloop()

# Освобождаем ресурсы после завершения работы
cap1.release()
cap2.release()
cv2.destroyAllWindows()