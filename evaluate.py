import os
import argparse
import time
import cv2
import numpy as np
import scipy.stats as stats
import pandas as pd
import matplotlib.pyplot as plt
from glob import glob
from skimage.measure import shannon_entropy
from skimage.metrics import structural_similarity as ssim
from bhe2pl import bhe2pl


def calculate_single_image_metrics(img):
    contrast = np.std(img, dtype=np.float64)
    entropy = shannon_entropy(img)
    lap_var = cv2.Laplacian(img, cv2.CV_64F).var()
    return {
        'Contrast': contrast,
        'Entropy': entropy,
        'LaplacianVar': lap_var,
    }


def calculate_metrics(img_orig, img_enh):
    # AMBE
    ambe = abs(np.mean(img_orig, dtype=np.float64) - np.mean(img_enh, dtype=np.float64))
    
    # PSNR (OpenCV)
    psnr = cv2.PSNR(img_orig, img_enh)
    
    # Contraste (Desviación Estándar)
    contrast = np.std(img_enh, dtype=np.float64)
    
    # Entropía (Shannon)
    entropy = shannon_entropy(img_enh)

    # MSE
    mse = np.mean((img_orig.astype(np.float64) - img_enh.astype(np.float64)) ** 2)

    # SSIM
    ssim_val = ssim(img_orig, img_enh, data_range=255)

    # Nitidez estimada por varianza del Laplaciano
    lap_var = cv2.Laplacian(img_enh, cv2.CV_64F).var()

    return {
        'AMBE': ambe,
        'PSNR': psnr,
        'Contrast': contrast,
        'Entropy': entropy,
        'MSE': mse,
        'SSIM': ssim_val,
        'LaplacianVar': lap_var,
    }


def apply_techniques(img, clahe_obj):
    start = time.perf_counter()
    he_img = cv2.equalizeHist(img)
    he_time_ms = (time.perf_counter() - start) * 1000.0

    start = time.perf_counter()
    clahe_img = clahe_obj.apply(img)
    clahe_time_ms = (time.perf_counter() - start) * 1000.0

    start = time.perf_counter()
    bhe_img = bhe2pl(img)
    bhe_time_ms = (time.perf_counter() - start) * 1000.0

    return {
        'HE': {'image': he_img, 'time_ms': he_time_ms},
        'CLAHE': {'image': clahe_img, 'time_ms': clahe_time_ms},
        'BHE2PL': {'image': bhe_img, 'time_ms': bhe_time_ms},
    }


def _save_reference_images(images_by_name, output_dir):
    cv2.imwrite(os.path.join(output_dir, 'referencia_original.png'), images_by_name['Original'])
    cv2.imwrite(os.path.join(output_dir, 'referencia_he.png'), images_by_name['HE'])
    cv2.imwrite(os.path.join(output_dir, 'referencia_clahe.png'), images_by_name['CLAHE'])
    cv2.imwrite(os.path.join(output_dir, 'referencia_bhe2pl.png'), images_by_name['BHE2PL'])


def _plot_histograms_grid(images_by_name, save_path):
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    ordered_names = ['Original', 'HE', 'CLAHE', 'BHE2PL']

    for ax, name in zip(axes.flat, ordered_names):
        hist = cv2.calcHist([images_by_name[name]], [0], None, [256], [0, 256]).flatten()
        ax.plot(hist, linewidth=1.2)
        ax.set_title(f'Histograma - {name}')
        ax.set_xlim(0, 255)
        ax.set_xlabel('Intensidad')
        ax.set_ylabel('Frecuencia')
        ax.grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close(fig)


def plot_reference_results(reference_image_path, output_dir, clahe_obj):
    ref_img = cv2.imread(reference_image_path, cv2.IMREAD_GRAYSCALE)
    if ref_img is None:
        print(f"No se pudo leer la imagen de referencia: {reference_image_path}")
        return

    processed = apply_techniques(ref_img, clahe_obj)
    images_by_name = {
        'Original': ref_img,
        'HE': processed['HE']['image'],
        'CLAHE': processed['CLAHE']['image'],
        'BHE2PL': processed['BHE2PL']['image'],
    }

    _save_reference_images(images_by_name, output_dir)
    _plot_histograms_grid(images_by_name, os.path.join(output_dir, 'referencia_histogramas.png'))


def save_metric_boxplots(df, output_dir):
    metrics = ['AMBE', 'PSNR', 'Contrast', 'Entropy', 'SSIM', 'MSE']
    techniques = ['Original', 'HE', 'CLAHE', 'BHE2PL']

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    for ax, metric in zip(axes.flat, metrics):
        data = []
        labels = []
        for tech in techniques:
            values = df[df['technique'] == tech][metric].dropna().values
            if len(values) > 0:
                data.append(values)
                labels.append(tech)

        if not data:
            ax.set_visible(False)
            continue

        ax.boxplot(data, tick_labels=labels, showmeans=True)
        ax.set_title(metric)
        ax.grid(axis='y', alpha=0.25)

    fig.suptitle('Distribución de métricas por técnica', fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'metricas_boxplot.png'), dpi=200, bbox_inches='tight')
    plt.close(fig)


def save_time_plot(df, output_dir):
    mean_times = df.groupby('technique', as_index=False)['Time_ms'].mean()
    mean_times = mean_times.dropna(subset=['Time_ms'])
    mean_times = mean_times.sort_values('Time_ms', ascending=True)

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(mean_times['technique'], mean_times['Time_ms'])
    ax.set_title('Tiempo promedio de procesamiento por técnica')
    ax.set_ylabel('Tiempo (ms)')
    ax.grid(axis='y', alpha=0.25)

    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f'{height:.2f}',
            ha='center',
            va='bottom',
            fontsize=9,
        )

    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'tiempos_promedio.png'), dpi=200, bbox_inches='tight')
    plt.close(fig)


def print_global_averages(summary):
    metrics = [
        ('AMBE', 'AMBE'),
        ('PSNR', 'PSNR'),
        ('Contrast', 'Contrast'),
        ('Entropy', 'Entropy'),
        ('MSE', 'MSE'),
        ('SSIM', 'SSIM'),
        ('LaplacianVar', 'LaplacianVar'),
        ('Time_ms', 'Time_ms'),
    ]

    print("\n--- Promedios Globales (linea por linea) ---")
    for method in summary.index:
        print(f"\nMetodo: {method}")
        for metric_name, mean_col in metrics:
            mean_val = summary.loc[method, mean_col]
            if pd.isna(mean_val):
                print(f"  {metric_name}: mean = N/A")
            else:
                print(f"  {metric_name}: mean = {mean_val:.6f}")
    

def evaluate_dataset(image_dir="dataset", output_dir="output", reference_image_path="dataset/referencia.jpg"):
    os.makedirs(output_dir, exist_ok=True)
    image_paths = glob(os.path.join(image_dir, '*.*'))
    # Filtro de formatos
    image_paths = [p for p in image_paths if p.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
    image_paths = sorted(image_paths)
    
    if not image_paths:
        print(f"No se encontraron imágenes en {image_dir}")
        return
    
    total_images = len(image_paths)
    results = []
    
    clahe_obj = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    # Genera plots solicitados para la imagen de referencia
    plot_reference_results(reference_image_path, output_dir, clahe_obj)
    
    for i, path in enumerate(image_paths, start=1):
        print(f"Evaluando imagen {i}/{total_images}")
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue

        original_metrics = calculate_single_image_metrics(img)
        results.append({
            'filename': os.path.basename(path),
            'technique': 'Original',
            'Time_ms': np.nan,
            'AMBE': np.nan,
            'PSNR': np.nan,
            'Contrast': original_metrics['Contrast'],
            'Entropy': original_metrics['Entropy'],
            'MSE': np.nan,
            'SSIM': np.nan,
            'LaplacianVar': original_metrics['LaplacianVar'],
        })

        processed = apply_techniques(img, clahe_obj)

        for tech_name in ['HE', 'CLAHE', 'BHE2PL']:
            enhanced = processed[tech_name]['image']
            metrics = calculate_metrics(img, enhanced)
            row = {
                'filename': os.path.basename(path),
                'technique': tech_name,
                'Time_ms': processed[tech_name]['time_ms'],
            }
            row.update(metrics)
            results.append(row)
        
    df = pd.DataFrame(results)

    if df.empty:
        print("No se pudieron procesar imágenes válidas.")
        return

    # Guarda resultados por imagen/técnica
    detailed_csv = os.path.join(output_dir, 'resultados_detallados.csv')
    df.to_csv(detailed_csv, index=False)

    summary = df.groupby('technique', as_index=True).agg({
        'AMBE': 'mean',
        'PSNR': 'mean',
        'Contrast': 'mean',
        'Entropy': 'mean',
        'MSE': 'mean',
        'SSIM': 'mean',
        'LaplacianVar': 'mean',
        'Time_ms': 'mean',
    })
    summary_csv = os.path.join(output_dir, 'resumen_metricas.csv')
    summary.to_csv(summary_csv)
    
    print_global_averages(summary)
    
    print("\n--- Pruebas Estadísticas (p-valor) ---")
    print("T-test pareado entre técnicas (HE, CLAHE, BHE2PL)")

    df_processed = df[df['technique'].isin(['HE', 'CLAHE', 'BHE2PL'])]
    pivot_metrics = {
        metric: df_processed.pivot(index='filename', columns='technique', values=metric).dropna()
        for metric in ['AMBE', 'PSNR', 'Contrast', 'Entropy', 'MSE', 'SSIM', 'LaplacianVar', 'Time_ms']
    }

    comparisons = [('HE', 'CLAHE'), ('HE', 'BHE2PL'), ('CLAHE', 'BHE2PL')]
    stats_rows = []

    for metric, pivot_df in pivot_metrics.items():
        for t1, t2 in comparisons:
            if t1 not in pivot_df.columns or t2 not in pivot_df.columns or len(pivot_df) < 2:
                continue
            t_stat, p_val = stats.ttest_rel(pivot_df[t1], pivot_df[t2], nan_policy='omit')
            sig = "Significativo" if p_val < 0.05 else "NO Significativo"
            print(f"{metric} ({t1} vs {t2}): p-value = {p_val:.4e} ({sig})")
            stats_rows.append({
                'Metric': metric,
                'Comparison': f'{t1} vs {t2}',
                't_stat': t_stat,
                'p_value': p_val,
                'Significance': sig,
            })

    if stats_rows:
        pd.DataFrame(stats_rows).to_csv(os.path.join(output_dir, 'pruebas_estadisticas.csv'), index=False)

    save_metric_boxplots(df, output_dir)
    save_time_plot(df, output_dir)

    print("\nArchivos generados en output:")
    print("- resultados_detallados.csv")
    print("- resumen_metricas.csv")
    print("- pruebas_estadisticas.csv (si hubo datos suficientes)")
    print("- referencia_original.png")
    print("- referencia_he.png")
    print("- referencia_clahe.png")
    print("- referencia_bhe2pl.png")
    print("- referencia_histogramas.png")
    print("- metricas_boxplot.png")
    print("- tiempos_promedio.png")
        
    return df

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluación de técnicas de mejora de contraste')
    parser.add_argument('--image-dir', default='dataset', help='Carpeta con imágenes de evaluación')
    parser.add_argument('--output-dir', default='output', help='Carpeta donde se guardarán resultados')
    parser.add_argument(
        '--reference-image',
        default=os.path.join('dataset', 'referencia.jpg'),
        help='Ruta de imagen de referencia para plots cualitativos',
    )
    args = parser.parse_args()

    evaluate_dataset(
        image_dir=args.image_dir,
        output_dir=args.output_dir,
        reference_image_path=args.reference_image,
    )
