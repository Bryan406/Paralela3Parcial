import cv2
import numpy as np
import multiprocessing as mp
import os
import time
from queue import Empty
import psutil
import matplotlib.pyplot as plt

def aplicar_filtro_bn(fragmento):
    return cv2.cvtColor(fragmento, cv2.COLOR_BGR2GRAY)

def procesar_parte(tareas, resultados, id_proceso, tareas_totales, output_dir):
    while True:
        try:
            tarea = tareas.get(timeout=0.05)
        except Empty:
            if tareas_totales.value == 0:
                return
            time.sleep(0.05)
            continue
        
        img_idx, frag_idx, fragmento = tarea
        resultado = aplicar_filtro_bn(fragmento)
        resultados.put((img_idx, frag_idx, resultado))
        with tareas_totales.get_lock():
            tareas_totales.value -= 1

def balanceo_distribuido(img_dir):
    imagenes = [os.path.join(img_dir, img) for img in os.listdir(img_dir) if img.endswith(('.png', '.jpg', '.jpeg'))]
    if not imagenes:
        print("❌ No se encontraron imágenes en la carpeta.")
        return
    
    output_dir = "imagenes_procesadas"
    os.makedirs(output_dir, exist_ok=True)
    
    num_procesos = min(mp.cpu_count(), len(imagenes) * 4)
    
    tareas = mp.Queue()
    resultados = mp.Queue()
    tareas_totales = mp.Value('i', 0)
    
    fragmentos_por_imagen = 8
    for img_idx, img_path in enumerate(imagenes):
        imagen = cv2.imread(img_path)
        if imagen is None:
            continue
        fragmentos = np.array_split(imagen, fragmentos_por_imagen, axis=0)
        for frag_idx, fragmento in enumerate(fragmentos):
            tareas.put((img_idx, frag_idx, fragmento))
            tareas_totales.value += 1
    
    procesos = [mp.Process(target=procesar_parte, args=(tareas, resultados, i, tareas_totales, output_dir)) for i in range(num_procesos)]
    for p in procesos:
        p.start()
    
    resultados_dict = {}
    while tareas_totales.value > 0 or not resultados.empty():
        try:
            img_idx, frag_idx, frag = resultados.get(timeout=2)
            if img_idx not in resultados_dict:
                resultados_dict[img_idx] = []
            resultados_dict[img_idx].append((frag_idx, frag))
        except Empty:
            pass
    
    for p in procesos:
        p.join()
    
    for img_idx in resultados_dict:
        resultados_dict[img_idx].sort(key=lambda x: x[0])
        fragmentos = [frag for _, frag in resultados_dict[img_idx]]
        imagen_final = np.concatenate(fragmentos, axis=0)
        cv2.imwrite(f"{output_dir}/imagen_bn_{img_idx}.png", imagen_final)
    
    print("✅ Procesamiento completado. Imágenes guardadas en 'imagenes_procesadas'.")

if __name__ == "__main__":
    mp.freeze_support()
    ruta = "C:\\Users\\rquis_9zzy7zj\\OneDrive - UNIVERSIDAD DE LAS FUERZAS ARMADAS ESPE\\Escritorio\\Paralela\\Tarea1Parcial3\\imagenes"
    balanceo_distribuido(ruta)
