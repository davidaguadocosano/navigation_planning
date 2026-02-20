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
    """Extrae la secuencia de nodos visitados del tensor de acciones"""
    raw_tour = actions[0, :, 0].cpu().numpy().astype(int)
    tour = []
    if len(raw_tour) > 0:
        tour.append(raw_tour[0])
        for i in range(1, len(raw_tour)):
            if raw_tour[i] != raw_tour[i-1]:
                tour.append(raw_tour[i])
    return np.array(tour)

def calculate_dist(nodes, tour):
    """Calcula la distancia euclidiana total cerrando el tour"""
    if len(tour) < 2: return 0.0
    coords = nodes[tour]
    coords_closed = np.vstack([coords, coords[0]])
    return np.sum(np.sqrt(np.sum(np.diff(coords_closed, axis=0)**2, axis=1)))

def visualize_comparison():
    # 1. Extraer paths de modelos
    load_path_2 = None
    if "--load_path_2" in sys.argv:
        idx = sys.argv.index("--load_path_2")
        load_path_2 = sys.argv[idx + 1]
        sys.argv.pop(idx); sys.argv.pop(idx)
    
    if not load_path_2:
        print("[!] Error: Indica --load_path_2")
        return

    # 2. Configuración y Aleatoriedad
    opts = get_options()
    seed = int(time.time())
    torch.manual_seed(seed); np.random.seed(seed)
    print(f"[*] Evaluando mapa con semilla: {seed}")

    opts.device = torch.device("cuda" if opts.use_cuda and torch.cuda.is_available() else "cpu")

    # 3. Cargar modelos
    model1, _, _ = load_model(opts=opts, train=False)
    original_path = opts.load_path
    opts.load_path = load_path_2
    model2, _, _ = load_model(opts=opts, train=False)
    opts.load_path = original_path
    model1.to(opts.device).eval(); model2.to(opts.device).eval()

    # 4. Generar problema común
    dataloader = get_generator(opts.env, 1, opts.num_nodes, opts.num_obs, opts.image_size, 1)
    batch = next(iter(dataloader))
    batch = move_to(batch, opts.device)
    nodes_np = batch['nodes'][0].cpu().numpy()

    # 5. Obtener soluciones y recompensas
    with torch.no_grad():
        res1 = model1(batch)
        res2 = model2(batch)
        
        t1 = process_tour(res1[2]) # Acciones en res[2]
        t2 = process_tour(res2[2])
        
        # Score positivo = mejor. Invertimos el signo del acumulado
        score1 = -res1[0].item() 
        score2 = -res2[0].item()

    # 6. Calcular Distancias
    d1 = calculate_dist(nodes_np, t1)
    d2 = calculate_dist(nodes_np, t2)

    # 7. Crear Títulos
    title_1 = f"Base | Nodos: {len(np.unique(t1))} | Dist: {d1:.3f} | Score: {score1:.1f}"
    title_2 = f"Pre | Nodos: {len(np.unique(t2))} | Dist: {d2:.3f} | Score: {score2:.1f}"

    # 8. Graficar AMBAS soluciones
    # Usamos el parámetro 'title' que acabamos de añadir a plot_utils
    plot_tsp_solution(batch['nodes'][0], t1, "eval_modelo_1.png", title=title_1)
    plot_tsp_solution(batch['nodes'][0], t2, "eval_modelo_2.png", title=title_2)
    
    print(f"\n[*] COMPARATIVA FINALIZADA:")
    print(f"    - Modelo 1: {title_1}")
    print(f"    - Modelo 2: {title_2}")

if __name__ == "__main__":
    visualize_comparison()