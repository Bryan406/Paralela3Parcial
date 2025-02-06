import cv2
import numpy as np
import multiprocessing as mp
import time
import psutil
import os

def aplicar_filtro_bn(fragmento):
    """
    Convierte un fragmento de la imagen a blanco y negro.
    """
    return cv2.cvtColor(fragmento, cv2.COLOR_BGR2GRAY)

def procesar_parte(args):
    """
    Procesa un fragmento aplicando el filtro en blanco y negro.
    """
    indice, fragmento = args
    resultado = aplicar_filtro_bn(fragmento)
    return indice, resultado  # Retorna el resultado sin tiempo

def balanceo_centralizado(img_path, output_dir):
    """
    Implementa balanceo centralizado con nodo maestro.
    """
    print(f"🔵 Nodo maestro: Cargando imagen {img_path}...")

    imagen = cv2.imread(img_path)
    if imagen is None:
        print(f"❌ Error: No se pudo cargar la imagen {img_path}. Verifica la ruta.")
        return

    height, width, _ = imagen.shape
    num_procesos = min(mp.cpu_count(), height)  # No más procesos que filas

    print(f"📊 Procesadores disponibles: {mp.cpu_count()}")
    print(f"🔄 Dividiendo imagen en {num_procesos} fragmentos...")

    # Dividir la imagen en fragmentos según el número de procesos
    fragmentos = np.array_split(imagen, num_procesos, axis=0)
    tareas = [(i, fragmento) for i, fragmento in enumerate(fragmentos)]

    # Iniciar los procesos en paralelo
    with mp.Pool(processes=num_procesos) as pool:
        resultados = pool.map(procesar_parte, tareas)

        # Cerrar el pool después de las tareas
        pool.close()
        pool.join()

    # Recopilando resultados
    print("📥 Nodo maestro: Recopilando resultados...")
    resultados.sort()  # Asegura el orden correcto
    imagen_final = np.vstack([frag for _, frag in resultados])

    # Convertir la imagen final de blanco y negro a BGR para guardarla correctamente
    imagen_final_bgr = cv2.cvtColor(imagen_final, cv2.COLOR_GRAY2BGR)

    # Ruta de salida
    output_path = os.path.join(output_dir, os.path.basename(img_path))
    cv2.imwrite(output_path, imagen_final_bgr)

    print(f"✅ Procesamiento completado para {img_path}. Imagen guardada como '{output_path}'.")

def procesar_imagenes_en_carpeta(carpeta_entrada, carpeta_salida):
    """
    Procesa todas las imágenes en la carpeta de entrada y guarda los resultados en la carpeta de salida.
    """
    if not os.path.exists(carpeta_salida):
        os.makedirs(carpeta_salida)
    
    # Obtener las rutas de todas las imágenes en la carpeta
    archivos = [f for f in os.listdir(carpeta_entrada) if f.lower().endswith(('jpg', 'jpeg', 'png', 'bmp'))]
    imagenes = [os.path.join(carpeta_entrada, archivo) for archivo in archivos]

    # Iniciar el tiempo total de procesamiento
    inicio_total = time.time()

    for img_path in imagenes:
        balanceo_centralizado(img_path, carpeta_salida)

    # Calcular el tiempo total de la tarea
    tiempo_total = time.time() - inicio_total
    print(f"⏰ Tiempo total de procesamiento para todas las imágenes: {tiempo_total:.4f} segundos.")

    # Mostrar estadísticas de rendimiento
    uso_cpu = psutil.cpu_percent(interval=1)
    print(f"💻 Uso de CPU durante el procesamiento: {uso_cpu}%")

if __name__ == "__main__":
    # Definir las rutas
    carpeta_entrada = r"C:\Users\Bryan406\Desktop\Paralela\Tarea1Parcial3\Paralela3Parcial\Tarea1Parcial3\imagenes"
    carpeta_salida = r"C:\Users\Bryan406\Desktop\Paralela\Tarea1Parcial3\Paralela3Parcial\Tarea1Parcial3\imagene_nodo_maestro"

    # Procesar todas las imágenes de la carpeta de entrada
    procesar_imagenes_en_carpeta(carpeta_entrada, carpeta_salida)
