import cv2
import numpy as np
import multiprocessing as mp

def aplicar_desenfoque_radial(fragmento):
    """
    Aplica un desenfoque gaussiano a un fragmento de la imagen.

    Parámetros:
        fragmento (numpy.ndarray): Fragmento de la imagen al que se le aplicará el desenfoque.

    Retorna:
        numpy.ndarray: Fragmento con el desenfoque gaussiano aplicado.
    """
    # Aplica un desenfoque gaussiano con un kernel de 15x15
    return cv2.GaussianBlur(fragmento, (15, 15), 0)

def procesar_parte(args):
    """
    Procesa un fragmento de la imagen aplicando el desenfoque radial.

    Parámetros:
        args (tuple): Tupla con el índice del fragmento y el fragmento en sí.

    Retorna:
        tuple: Tupla con el índice del fragmento y el resultado procesado.
    """
    indice, fragmento = args
    resultado = aplicar_desenfoque_radial(fragmento)  # Aplica el desenfoque
    return indice, resultado

def balanceo_centralizado(img_path):
    """
    Realiza el procesamiento paralelo de una imagen en fragmentos, aplicando un desenfoque radial 
    a cada fragmento y luego reconstruyendo la imagen final.

    Parámetros:
        img_path (str): Ruta del archivo de la imagen que se procesará.

    Este método divide la imagen en fragmentos, distribuye los fragmentos entre varios procesos 
    para ser procesados en paralelo y luego reconstruye la imagen final procesada.
    """
    # Leer la imagen desde la ruta proporcionada
    imagen = cv2.imread(img_path)
    
    # Obtener las dimensiones de la imagen
    height, width, _ = imagen.shape
    
    # Determinar el número de procesos disponibles en el sistema
    num_procesos = mp.cpu_count()

    # Dividir la imagen en fragmentos (una para cada proceso)
    fragmentos = np.array_split(imagen, num_procesos, axis=0)

    # Crear una lista de tareas, donde cada tarea es un fragmento y su índice
    tareas = [(i, fragmento) for i, fragmento in enumerate(fragmentos)]
    
    # Crear un pool de procesos para procesar las tareas en paralelo
    with mp.Pool(processes=num_procesos) as pool:
        fragmentos_procesados = pool.map(procesar_parte, tareas)

    # Mostrar el progreso en consola (no visualiza cada fragmento)
    for i, _ in sorted(fragmentos_procesados):
        print(f"Procesando fragmento {i + 1}/{len(fragmentos_procesados)}...")
    
    # Reconstruir la imagen final uniendo todos los fragmentos procesados
    imagen_final = np.vstack([frag for _, frag in sorted(fragmentos_procesados)])
    
    # Guardar la imagen procesada en un archivo
    cv2.imwrite("imagen_procesada.png", imagen_final)
    
    # Mostrar la imagen final procesada (solo al final)
    cv2.imshow("Imagen Final", imagen_final)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    """
    Función principal que ejecuta el procesamiento de la imagen con balanceo centralizado.

    Inicia el balanceo centralizado y aplica el desenfoque radial en paralelo.
    """
    # Soporte para Windows en multiprocessing
    mp.freeze_support()

    # Ruta del archivo de imagen a procesar
    balanceo_centralizado(r"C:\Users\rquis_9zzy7zj\OneDrive - UNIVERSIDAD DE LAS FUERZAS ARMADAS ESPE\Escritorio\Paralela\Tarea1Parcial3\pexels-sanaan-3052361.jpg")
