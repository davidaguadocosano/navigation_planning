import torch
import numpy as np
import sys
import time
from train import get_options
from nets.model import load_model
from envs import get_generator
from utils.plot_utils import plot_tsp_solution
from utils.train_utils import move_to

def process_tour(actions):
    """Extrae y limpia la ruta de nodos de las acciones del decoder"""
    # En ar_decoder.py, actions tiene forma [Batch, Pasos, 2]
    # La primera columna [:, 0] son los índices de los nodos
    raw_tour = actions[0, :, 0].cpu().numpy().astype(int)
    
    # Filtro para navegación: eliminamos repeticiones consecutivas
    tour = []
    if len(raw_tour) > 0:
        tour.append(raw_tour[0])
        for i in range(1, len(raw_tour)):
            if raw_tour[i] != raw_tour[i-1]:
                tour.append(raw_tour[i])
    return np.array(tour)

def visualize_comparison():
    # 1. TRUCO: Extraemos --load_path_2 antes de llamar a get_options
    # para evitar el error de 'unrecognized arguments'
    load_path_2 = None
    if "--load_path_2" in sys.argv:
        idx = sys.argv.index("--load_path_2")
        load_path_2 = sys.argv[idx + 1]
        # Eliminamos estos dos elementos de sys.argv para que get_options no los vea
        sys.argv.pop(idx) # Elimina --load_path_2
        sys.argv.pop(idx) # Elimina el valor del path
    
    if not load_path_2:
        print("[!] Error: Debes proporcionar --load_path_2 para comparar.")
        return

    # 2. Configurar opciones base
    opts = get_options()
    # FORZAR ALEATORIEDAD:
    # Si no pasas una semilla por el terminal, usamos el tiempo actual
    # Esto asegura que el mapa sea distinto en cada ejecución
    new_seed = int(time.time())
    torch.manual_seed(new_seed)
    np.random.seed(new_seed)
    print(f"[*] Generando mapa con nueva semilla: {new_seed}")
    
    opts.device = torch.device("cuda" if opts.use_cuda and torch.cuda.is_available() else "cpu")

    # 3. Cargar ambos modelos
    print(f"[*] Cargando Modelo 1: {opts.load_path}")
    model1, _, _ = load_model(opts=opts, train=False)
    
    print(f"[*] Cargando Modelo 2: {load_path_2}")
    original_path = opts.load_path
    opts.load_path = load_path_2
    model2, _, _ = load_model(opts=opts, train=False)
    opts.load_path = original_path # Restauramos por si acaso

    model1.to(opts.device).eval()
    model2.to(opts.device).eval()

    # 4. Generar UN SOLO problema idéntico (fijamos semilla)
    #torch.manual_seed(1234)        descomentar si quiero usar siempre el mismo escenario
    dataloader = get_generator(opts.env, 1, opts.num_nodes, opts.num_obs, opts.image_size, 1)
    batch = next(iter(dataloader))
    batch = move_to(batch, opts.device)

    # 5. Obtener soluciones
    with torch.no_grad():
        out1 = model1(batch)
        out2 = model2(batch)
        
        # ar_decoder.py devuelve (rewards, log_probs, actions, success)
        # Las acciones (el tour) están en el índice 2
        tour1 = process_tour(out1[2])
        tour2 = process_tour(out2[2])

    print(f"[*] Modelo 1 - Nodos únicos visitados: {len(np.unique(tour1))}")
    print(f"[*] Modelo 2 - Nodos únicos visitados: {len(np.unique(tour2))}")

    # 6. Guardar imágenes comparativas
    plot_tsp_solution(batch['nodes'][0], tour1, "comparison_baseline.png")
    plot_tsp_solution(batch['nodes'][0], tour2, "comparison_pretrained.png")
    
    print("[*] Imágenes generadas con éxito.")

if __name__ == "__main__":
    visualize_comparison()