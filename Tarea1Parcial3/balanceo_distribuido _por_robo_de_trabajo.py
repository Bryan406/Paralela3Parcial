import cv2
import numpy as np
import multiprocessing as mp
import os
import time
import random
import psutil  # Para la carga de la CPU

def aplicar_filtro_bn(fragmento):
    return cv2.cvtColor(fragmento, cv2.COLOR_BGR2GRAY)

def procesar_parte(tareas, resultados, id_proceso, tareas_totales, tareas_realizadas, robos, colas_procesos):
    while True:
        try:
            tarea = colas_procesos[id_proceso].get(timeout=0.05)  # Intentar obtener una tarea propia
        except:
            # Intentar robar tarea de otro proceso
            otros_procesos = list(range(len(colas_procesos)))
            random.shuffle(otros_procesos)
            for otro in otros_procesos:
                if otro != id_proceso:
                    try:
                        tarea = colas_procesos[otro].get(timeout=0.05)
                        with robos.get_lock():
                            robos[id_proceso] += 1
                        print(f"💀 Proceso {id_proceso} robó tarea de Proceso {otro}")
                        break
                    except:
                        continue
            else:
                if tareas_totales.value == 0:
                    return
                time.sleep(0.05)
                continue
        
        img_idx, frag_idx, fragmento = tarea
        resultado = aplicar_filtro_bn(fragmento)
        resultados.put((img_idx, frag_idx, resultado))
        
        with tareas_totales.get_lock():
            tareas_totales.value -= 1
        
        with tareas_realizadas.get_lock():
            tareas_realizadas[id_proceso] += 1
        
        print(f"🛠️ Proceso {id_proceso} completó tarea {frag_idx} de imagen {img_idx}")

def balanceo_work_stealing(img_dir):
    imagenes = [os.path.join(img_dir, img) for img in os.listdir(img_dir) if img.endswith(('.png', '.jpg', '.jpeg'))]
    if not imagenes:
        print("❌ No se encontraron imágenes en la carpeta.")
        return
    
    output_dir = "imagenes_procesadas"
    os.makedirs(output_dir, exist_ok=True)
    
    num_procesos = min(mp.cpu_count(), len(imagenes) * 4)
    tareas_totales = mp.Value('i', 0)
    tareas_realizadas = mp.Array('i', [0] * num_procesos)
    robos = mp.Array('i', [0] * num_procesos)
    resultados = mp.Manager().Queue()
    colas_procesos = [mp.Queue() for _ in range(num_procesos)]
    
    fragmentos_por_imagen = 8
    for img_idx, img_path in enumerate(imagenes):
        imagen = cv2.imread(img_path)
        if imagen is None:
            continue
        fragmentos = np.array_split(imagen, fragmentos_por_imagen, axis=0)
        for frag_idx, fragmento in enumerate(fragmentos):
            colas_procesos[random.randint(0, num_procesos - 1)].put((img_idx, frag_idx, fragmento))
            tareas_totales.value += 1
    
    procesos = [mp.Process(target=procesar_parte, args=(None, resultados, i, tareas_totales, tareas_realizadas, robos, colas_procesos)) for i in range(num_procesos)]
    for p in procesos:
        p.start()
    
    resultados_dict = {}
    while tareas_totales.value > 0 or not resultados.empty():
        try:
            img_idx, frag_idx, frag = resultados.get(timeout=2)
            if img_idx not in resultados_dict:
                resultados_dict[img_idx] = []
            resultados_dict[img_idx].append((frag_idx, frag))
        except:
            pass
    
    for p in procesos:
        p.join()
    
    for img_idx in resultados_dict:
        resultados_dict[img_idx].sort(key=lambda x: x[0])
        fragmentos = [frag for _, frag in resultados_dict[img_idx]]
        imagen_final = np.concatenate(fragmentos, axis=0)
        cv2.imwrite(f"{output_dir}/imagen_bn_{img_idx}.png", imagen_final)
    
    print("✅ Procesamiento completado. Imágenes guardadas en 'imagenes_procesadas'.")
    
    print("\n📊 Resumen de tareas:")
    for i in range(num_procesos):
        print(f"🔹 Proceso {i}: {tareas_realizadas[i]} tareas completadas, {robos[i]} tareas robadas.")
    
    # Cargar la carga de la CPU después de que todo esté completo
    cpu_usage = psutil.cpu_percent(interval=1)  # Obtiene el uso promedio del CPU durante 1 segundo
    print(f"\n💻 Uso de CPU durante el procesamiento: {cpu_usage}%")

if __name__ == "__main__":
    mp.freeze_support()
    ruta = r"C:\\Users\\Bryan406\\Desktop\\Paralela\\Tarea1Parcial3\\Paralela3Parcial\\Tarea1Parcial3\\imagenes"
    inicio_total = time.time()
    balanceo_work_stealing(ruta)
    tiempo_total = time.time() - inicio_total
    print(f"\n⏰ Tiempo total de procesamiento: {tiempo_total:.4f} segundos.")
