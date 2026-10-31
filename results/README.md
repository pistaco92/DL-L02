# results

Aqui `main.py` escribe CSV, Markdown y PNG (por defecto `--output-dir results`).

Esos archivos no se suben a Git. Cada grupo genera los suyos con:

```bash
python main.py --data-path "dataset/15 atributos R0-R5.sav" --all-targets --epochs 5
```

Archivos generados:

| Archivo | Contenido |
|---|---|
| `resultados.csv` / `.md` | Tabla completa algoritmo x objetivo x metricas (media ± std externa) |
| `ranking_f1.csv` / `.md` | Orden por F1 macro (maximizar) |
| `ranking_mae.csv` / `.md` | Orden por MAE ordinal (minimizar) |
| `ranking_<metrica>.*` | Ranking de `--rank-metric` |
| `seleccion_hp.csv` | Configuracion elegida por la busqueda interna en cada fold externo |
| `barras_f1.png`, `barras_mae.png` | Barras por objetivo y algoritmo |
| `ranking_<metrica>.png` | Barras del ranking activo |
| `confusion_<algoritmo>_<target>.png` | Matriz de confusion del ultimo fold externo |
