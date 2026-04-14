import os
import argparse
import time
from itertools import combinations
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
    return {
        'Contrast': contrast,
        'Entropy': entropy,
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

    # SSIM
    ssim_val = ssim(img_orig, img_enh, data_range=255)

    return {
        'AMBE': ambe,
        'PSNR': psnr,
        'Contrast': contrast,
        'Entropy': entropy,
        'SSIM': ssim_val,
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
        ax.plot(hist, linewidth=1.2, color='black')
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
    metrics = ['AMBE', 'PSNR', 'Contrast', 'Entropy', 'SSIM']
    techniques = ['Original', 'HE', 'CLAHE', 'BHE2PL']

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    flat_axes = axes.flat
    for ax, metric in zip(flat_axes, metrics):
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

        boxprops = dict(linestyle='-', linewidth=1, color='black')
        medianprops = dict(linestyle='-', linewidth=1.5, color='black')
        meanprops = dict(marker='o', markeredgecolor='black', markerfacecolor='black')
        whiskerprops = dict(color='black')
        capprops = dict(color='black')
        
        ax.boxplot(data, tick_labels=labels, showmeans=True, 
                   boxprops=boxprops, medianprops=medianprops, meanprops=meanprops, 
                   whiskerprops=whiskerprops, capprops=capprops)
        ax.set_title(metric)
        ax.grid(axis='y', alpha=0.25)

    for ax in list(axes.flat)[len(metrics):]:
        ax.set_visible(False)

    fig.suptitle('Distribución de métricas por técnica', fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'metricas_boxplot.png'), dpi=200, bbox_inches='tight')
    plt.close(fig)


def save_time_plot(df, output_dir):
    mean_times = df.groupby('technique', as_index=False)['Time_ms'].mean()
    mean_times = mean_times.dropna(subset=['Time_ms'])
    mean_times = mean_times.sort_values('Time_ms', ascending=True)

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(mean_times['technique'], mean_times['Time_ms'], color='gray', edgecolor='black')
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
        ('AMBE', 'AMBE_mean', 'AMBE_std'),
        ('PSNR', 'PSNR_mean', 'PSNR_std'),
        ('Contrast', 'Contrast_mean', 'Contrast_std'),
        ('Entropy', 'Entropy_mean', 'Entropy_std'),
        ('SSIM', 'SSIM_mean', 'SSIM_std'),
        ('Time_ms', 'Time_ms_mean', 'Time_ms_std'),
    ]

    print("\n--- Promedios Globales (linea por linea) ---")
    for method in summary.index:
        print(f"\nMetodo: {method}")
        for metric_name, mean_col, std_col in metrics:
            mean_val = summary.loc[method, mean_col]
            std_val = summary.loc[method, std_col]
            if pd.isna(mean_val) or pd.isna(std_val):
                print(f"  {metric_name}: N/A")
            else:
                print(f"  {metric_name}: {mean_val:.3f} ± {std_val:.3f}")

    best_ambe = summary['AMBE_mean'].min()
    best_psnr = summary['PSNR_mean'].max()
    best_ssim = summary['SSIM_mean'].max()
    best_entr = summary['Entropy_mean'].max()
    best_cont = summary['Contrast_mean'].max()
    best_t = summary['Time_ms_mean'].min()
    
    # Export to LaTeX
    with open('output/tabla_resultados.tex', 'w', encoding='utf-8') as f:
        f.write('\\begin{tabular}{lcccccc}\n')
        f.write('\\toprule\n')
        f.write('Método & T (ms) & AMBE & PSNR & SSIM & Entr. & Cont. \\\\\n')
        f.write('\\midrule\n')
        for method in ['Original', 'HE', 'CLAHE', 'BHE2PL']:
            if method not in summary.index: continue
            
            def fmt(col_mean, col_std, is_best=False):
                val = summary.loc[method, col_mean]
                std = summary.loc[method, col_std]
                if pd.isna(val): return '--'
                res = f"{val:.3f} $\\pm$ {std:.3f}"
                if is_best: res = f"\\textbf{{{res}}}"
                return res

            t_str = fmt('Time_ms_mean', 'Time_ms_std', summary.loc[method, 'Time_ms_mean'] == best_t)
            ambe_str = fmt('AMBE_mean', 'AMBE_std', summary.loc[method, 'AMBE_mean'] == best_ambe)
            psnr_str = fmt('PSNR_mean', 'PSNR_std', summary.loc[method, 'PSNR_mean'] == best_psnr)
            ssim_str = fmt('SSIM_mean', 'SSIM_std', summary.loc[method, 'SSIM_mean'] == best_ssim)
            entr_str = fmt('Entropy_mean', 'Entropy_std', summary.loc[method, 'Entropy_mean'] == best_entr)
            cont_str = fmt('Contrast_mean', 'Contrast_std', summary.loc[method, 'Contrast_mean'] == best_cont)

            f.write(f"{method} & {t_str} & {ambe_str} & {psnr_str} & {ssim_str} & {entr_str} & {cont_str} \\\\\n")
        
        f.write('\\bottomrule\n')
        f.write('\\end{tabular}\n')


def _format_pvalue_latex(p_value):
    if pd.isna(p_value):
        return '--'
    if p_value == 0:
        return '$<1\\times10^{-300}$'
    exponent = int(np.floor(np.log10(abs(p_value))))
    mantissa = p_value / (10 ** exponent)
    return f"${mantissa:.2f}\\times10^{{{exponent}}}$"


def save_pvalues_latex_table(stats_rows, output_dir):
    if not stats_rows:
        return

    selected_metrics = ['AMBE', 'PSNR', 'SSIM', 'Time_ms']
    metric_order = {
        'AMBE': 0,
        'PSNR': 1,
        'SSIM': 2,
        'Time_ms': 3,
    }
    metric_display = {
        'AMBE': 'AMBE',
        'PSNR': 'PSNR',
        'SSIM': 'SSIM',
        'Time_ms': 'Tiempo',
    }

    stats_df = pd.DataFrame(stats_rows)
    stats_df = stats_df[stats_df['Metric'].isin(selected_metrics)].copy()
    if stats_df.empty:
        return

    stats_df = stats_df.sort_values(
        by=['Metric', 'Comparison'],
        key=lambda col: col.map(metric_order) if col.name == 'Metric' else col,
    )

    latex_path = os.path.join(output_dir, 'tabla_pvalores.tex')
    with open(latex_path, 'w', encoding='utf-8') as f:
        f.write('\\begin{tabular}{llcc}\n')
        f.write('\\toprule\n')
        f.write('Metrica & Comparacion & $p$-valor & Sig. \\\\\n')
        f.write('\\midrule\n')

        for _, row in stats_df.iterrows():
            metric = metric_display.get(row['Metric'], row['Metric'])
            comparison = row['Comparison']
            p_value = _format_pvalue_latex(row['p_value'])
            significance = 'Si' if row['p_value'] < 0.05 else 'No'
            f.write(f"{metric} & {comparison} & {p_value} & {significance} \\\\\n")

        f.write('\\bottomrule\n')
        f.write('\\end{tabular}\n')
    

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
            'SSIM': np.nan,
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
        'AMBE': ['mean', 'std'],
        'PSNR': ['mean', 'std'],
        'Contrast': ['mean', 'std'],
        'Entropy': ['mean', 'std'],
        'SSIM': ['mean', 'std'],
        'Time_ms': ['mean', 'std'],
    })
    summary.columns = ['_'.join(col).strip() for col in summary.columns.to_flat_index()]
    summary_csv = os.path.join(output_dir, 'resumen_metricas.csv')
    summary.to_csv(summary_csv)
    
    print_global_averages(summary)
    
    print("\n--- Pruebas Estadísticas (p-valor) ---")
    print("T-test pareado entre técnicas (HE, CLAHE, BHE2PL)")

    metric_techniques = {
        'AMBE': ['HE', 'CLAHE', 'BHE2PL'],
        'PSNR': ['HE', 'CLAHE', 'BHE2PL'],
        'Contrast': ['Original', 'HE', 'CLAHE', 'BHE2PL'],
        'Entropy': ['Original', 'HE', 'CLAHE', 'BHE2PL'],
        'SSIM': ['HE', 'CLAHE', 'BHE2PL'],
        'Time_ms': ['HE', 'CLAHE', 'BHE2PL'],
    }

    stats_rows = []

    for metric, techniques in metric_techniques.items():
        pivot_df = df[df['technique'].isin(techniques)].pivot(
            index='filename', columns='technique', values=metric
        )

        for t1, t2 in combinations(techniques, 2):
            if t1 not in pivot_df.columns or t2 not in pivot_df.columns or len(pivot_df) < 2:
                continue
            pair_df = pivot_df[[t1, t2]].dropna()
            if len(pair_df) < 2:
                continue

            t_stat, p_val = stats.ttest_rel(pair_df[t1], pair_df[t2], nan_policy='omit')
            sig = "Significativo" if p_val < 0.05 else "NO Significativo"
            print(f"{metric} ({t1} vs {t2}): {p_val:.4e}")
            stats_rows.append({
                'Metric': metric,
                'Comparison': f'{t1} vs {t2}',
                't_stat': t_stat,
                'p_value': p_val,
                'Significance': sig,
            })

    if stats_rows:
        pd.DataFrame(stats_rows).to_csv(os.path.join(output_dir, 'pruebas_estadisticas.csv'), index=False)
        save_pvalues_latex_table(stats_rows, output_dir)
        print("Pruebas estadísticas guardadas en output/pruebas_estadisticas.csv")

    save_metric_boxplots(df, output_dir)
    save_time_plot(df, output_dir)

    print("\nArchivos generados en output:")
    print("- resultados_detallados.csv")
    print("- resumen_metricas.csv")
    print("- pruebas_estadisticas.csv")
    print("- tabla_pvalores.tex")
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
