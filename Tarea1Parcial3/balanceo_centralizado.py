import cv2
import numpy as np
import multiprocessing as mp
import time
import psutil
import matplotlib.pyplot as plt

def aplicar_filtro_bn(fragmento):
    """
    Convierte un fragmento de la imagen a blanco y negro.
    """
    return cv2.cvtColor(fragmento, cv2.COLOR_BGR2GRAY)

def procesar_parte(args):
    """
    Procesa un fragmento aplicando el filtro en blanco y negro.
    También mide el tiempo de procesamiento.
    """
    indice, fragmento = args
    inicio = time.time()
    resultado = aplicar_filtro_bn(fragmento)
    fin = time.time()
    return indice, resultado, fin - inicio  # Retorna el tiempo tomado

def balanceo_centralizado(img_path):
    """
    Implementa balanceo centralizado con nodo maestro.
    """
    print("🔵 Nodo maestro: Cargando imagen...")
    imagen = cv2.imread(img_path)
    if imagen is None:
        print("❌ Error: No se pudo cargar la imagen. Verifica la ruta.")
        return

    height, width, _ = imagen.shape
    num_procesos = min(mp.cpu_count(), height)  # No más procesos que filas

    print(f"📊 Procesadores disponibles: {mp.cpu_count()}")
    print(f"🔄 Dividiendo imagen en {num_procesos} fragmentos...")

    fragmentos = np.array_split(imagen, num_procesos, axis=0)
    tareas = [(i, fragmento) for i, fragmento in enumerate(fragmentos)]

    tiempos_procesos = []

    with mp.Pool(processes=num_procesos) as pool:
        resultados = pool.map(procesar_parte, tareas)

    print("📥 Nodo maestro: Recopilando resultados...")
    resultados.sort()  # Asegura el orden correcto
    imagen_final = np.vstack([frag for _, frag, tiempo in resultados])

    # Guardar los tiempos de procesamiento por fragmento
    tiempos_procesos = [tiempo for _, _, tiempo in resultados]

    # Convertimos la imagen a BGR para guardarla correctamente
    imagen_final_bgr = cv2.cvtColor(imagen_final, cv2.COLOR_GRAY2BGR)
    cv2.imwrite("imagen_bn.png", imagen_final_bgr)
    
    print("✅ Procesamiento completado. Imagen guardada como 'imagen_bn.png'.")

    # Mostrar estadísticas de rendimiento
    uso_cpu = psutil.cpu_percent(interval=1)
    print(f"💻 Uso de CPU durante el procesamiento: {uso_cpu}%")

    # Graficar los tiempos de procesamiento
    plt.figure(figsize=(10, 5))
    plt.bar(range(num_procesos), tiempos_procesos, color='blue')
    plt.xlabel("ID del proceso")
    plt.ylabel("Tiempo de procesamiento (s)")
    plt.title("Tiempo de procesamiento por fragmento en balanceo centralizado")
    plt.show()


if __name__ == "__main__":
    mp.freeze_support()
    ruta = r"C:\Users\rquis_9zzy7zj\OneDrive - UNIVERSIDAD DE LAS FUERZAS ARMADAS ESPE\Escritorio\Paralela\Tarea1Parcial3\pexels-sanaan-3052361.jpg"
    balanceo_centralizado(ruta)
