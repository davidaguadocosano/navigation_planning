#no funciona
import os
from tensorboard.backend.event_processing import event_accumulator

# Tu ruta específica
log_dir = 'outputs/quick_baseline/tsp-graph_20/gtn_tsp-ar_2026-02-13-00-10-13/log_dir/'

# Buscamos el archivo de eventos
files = [f for f in os.listdir(log_dir) if 'events' in f]
if not files:
    print("No se encontraron archivos de eventos en esa carpeta.")
else:
    event_file = files[0]
    ea = event_accumulator.EventAccumulator(os.path.join(log_dir, event_file))
    ea.Reload()

    # Extraemos la recompensa de validación
    if 'val_avg_reward' in ea.Tags()['scalars']:
        print("\n--- RECOMPENSAS RECUPERADAS (BASELINE) ---")
        for e in ea.Scalars('val_avg_reward'):
            print(f"Época {e.step}: {e.value:.4f}")
    else:
        print(f"Tags disponibles: {ea.Tags()['scalars']}")