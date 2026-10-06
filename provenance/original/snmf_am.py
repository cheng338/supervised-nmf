import argparse
import glob
import os
import time
import warnings
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.decomposition import NMF, PCA
from sklearn.feature_selection import SelectPercentile, f_classif
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline

warnings.filterwarnings("ignore", category=UserWarning)


# ==========================================
# 1. CORE DATA INGESTION ENGINE
# ==========================================
def load_celeba_task(
    data_dir, attribute_name, N_samples=5000, img_size=(64, 64)
):
    attr_path = os.path.join(os.path.dirname(data_dir), "list_attr_celeba.csv")
    if not os.path.exists(attr_path):
        print(f"ERROR: Attribute lookup file missing at {attr_path}.")
        exit()

    df_attr = pd.read_csv(attr_path)
    df_attr[attribute_name] = df_attr[attribute_name].replace(-1, 0)
    label_dict = dict(zip(df_attr["image_id"], df_attr[attribute_name]))

    image_paths = sorted(glob.glob(os.path.join(data_dir, "*.jpg")))[
        :N_samples
    ]
    if len(image_paths) == 0:
        print(
            f"ERROR: Zero image files located within target repository: {data_dir}"
        )
        exit()

    X_list = []
    y_list = []

    for path in image_paths:
        filename = os.path.basename(path)
        if filename in label_dict:
            img = Image.open(path).convert("L").resize(img_size)
            X_list.append(np.array(img, dtype=np.float32).flatten())
            y_list.append(label_dict[filename])

    X = np.array(X_list, dtype=np.float32) / 255.0
    y = np.array(y_list, dtype=np.int64)

    return X, y


# ==========================================
# 2. FEATURE RECOVERY VISUALIZATION ENGINE
# ==========================================
def generate_feature_recovery_comparison_plots(
    X_train: np.ndarray,
    selector: SelectPercentile,
    nmf_std: NMF,
    nmf_snmf: NMF,
    attribute_name: str,
    img_size: tuple = (64, 64),
    n_display: int = 6,
):
    """Exports standalone comparison figures (PCA vs. Standard NMF vs. sNMF)

    demonstrating facial feature recovery and negative loading failure.
    """
    print(
        f"Exporting facial feature recovery proof images for: {attribute_name}..."
    )

    n_comp = min(n_display, nmf_std.n_components, nmf_snmf.n_components)

    # 1. PCA Eigenfaces (Demonstrating failure due to negative loadings / ghosting)
    pca = PCA(n_components=n_comp, random_state=42)
    pca.fit(X_train)
    H_pca = pca.components_

    # 2. Standard Unsupervised NMF Basis (Demonstrating background clutter capture)
    H_std = nmf_std.components_[:n_comp]

    # 3. Supervised NMF Basis (Reconstructed onto full canvas via zero-padding)
    H_snmf_compressed = nmf_snmf.components_[:n_comp]
    support_mask = selector.get_support()
    H_snmf_canvas = np.zeros((n_comp, X_train.shape[1]), dtype=np.float32)
    for k in range(n_comp):
        H_snmf_canvas[k, support_mask] = H_snmf_compressed[k]

    fig, axes = plt.subplots(3, n_comp, figsize=(2.8 * n_comp, 9.0))
    if n_comp == 1:
        axes = np.expand_dims(axes, axis=1)

    fig.suptitle(
        f"Facial Feature Recovery & Interpretability: {attribute_name}",
        fontsize=15,
        fontweight="bold",
        y=0.97,
    )

    row_labels = [
        "PCA\n(Fails due to Negativity)",
        "Standard NMF\n(Background Clutter)",
        "Supervised NMF\n(Recovers Features)",
    ]

    for k in range(n_comp):
        # Row 1: PCA with diverging colormap to expose negative values/ghosting
        vmax_val = np.max(np.abs(H_pca[k]))
        axes[0, k].imshow(
            H_pca[k].reshape(img_size),
            cmap="seismic",
            vmin=-vmax_val,
            vmax=vmax_val,
        )
        axes[0, k].set_title(f"Component {k+1}", fontsize=11)
        axes[0, k].axis("off")

        # Row 2: Standard NMF
        axes[1, k].imshow(
            H_std[k].reshape(img_size), cmap="viridis", vmin=0
        )
        axes[1, k].set_title(f"Component {k+1}", fontsize=11)
        axes[1, k].axis("off")

        # Row 3: Supervised NMF
        axes[2, k].imshow(
            H_snmf_canvas[k].reshape(img_size), cmap="viridis", vmin=0
        )
        axes[2, k].set_title(f"Component {k+1}", fontsize=11)
        axes[2, k].axis("off")

    for i, label in enumerate(row_labels):
        axes[i, 0].text(
            -0.22,
            0.5,
            label,
            transform=axes[i, 0].transAxes,
            fontsize=11,
            fontweight="bold",
            va="center",
            ha="right",
        )

    plt.tight_layout(rect=[0.18, 0.0, 1.0, 0.93])
    output_filename = f"Feature_Recovery_Comparison_{attribute_name}.png"
    plt.savefig(output_filename, dpi=300, bbox_inches="tight")
    plt.close()
    print(
        f" -> Feature recovery image saved successfully to: {output_filename}"
    )


# ==========================================
# 3. PROFILED BENCHMARK ENGINE (PERF_COUNTER)
# ==========================================
def run_evaluation_showdown(X, y, attribute_name):
    N_samples = X.shape[0]
    d_dim = X.shape[1]

    print("\n" + "=" * 105)
    print(f"TARGET ATTRIBUTE EVALUATION: {attribute_name.upper()}")
    print(f"Topography Setup: N = {N_samples}, d = {d_dim} | cv = 5")
    print("=" * 105)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # ---------------------------------------------------------
    # FRAMEWORK 1: Original KNN (Full Ambient Feature Space)
    # ---------------------------------------------------------
    knn_ambient = KNeighborsClassifier(n_neighbors=3, n_jobs=-1)
    screen_time_knn, mf_time_knn = 0.0000, 0.0000

    t0 = time.perf_counter()
    knn_ambient.fit(X_train, y_train)
    fit_time_knn = time.perf_counter() - t0

    t1 = time.perf_counter()
    preds_knn = knn_ambient.predict(X_test)
    test_time_knn = time.perf_counter() - t1
    acc_knn = accuracy_score(y_test, preds_knn)

    # ---------------------------------------------------------
    # OFFLINE TUNING PHASE: Exhaustive Co-Optimization (cv=5)
    # ---------------------------------------------------------
    tuning_pipeline = Pipeline([
        ("selector", SelectPercentile(score_func=f_classif)),
        (
            "nmf",
            NMF(
                init="nndsvda",
                solver="cd",
                max_iter=400,
                random_state=42,
            ),
        ),
        ("knn", KNeighborsClassifier(n_neighbors=3, n_jobs=-1)),
    ])

    # 169-Point Search Grid (K=200 removed)
    param_grid = {
        "selector__percentile": [
            1,
            2,
            5,
            10,
            15,
            20,
            25,
            30,
            40,
            50,
            60,
            70,
            80,
        ],
        "nmf__n_components": [
            2,
            5,
            10,
            15,
            20,
            30,
            40,
            50,
            60,
            80,
            100,
            120,
            150,
        ],
    }

    print(
        "Running Exhaustive 169-Point 2D Grid Search (P X K-Space | cv=5)..."
    )
    grid_search = GridSearchCV(
        tuning_pipeline, param_grid, cv=5, scoring="accuracy", n_jobs=-1
    )
    grid_search.fit(X_train, y_train)

    optimal_P = grid_search.best_params_["selector__percentile"]
    optimal_K = grid_search.best_params_["nmf__n_components"]
    print(f"Tuning Locked -> P: {optimal_P}%, K: {optimal_K}")

    # ---------------------------------------------------------
    # FRAMEWORK 3: Supervised NMF (sNMF -> Strictly Profiled)
    # ---------------------------------------------------------
    selector_snmf = SelectPercentile(score_func=f_classif, percentile=optimal_P)
    nmf_snmf = NMF(
        n_components=optimal_K,
        init="nndsvda",
        solver="cd",
        max_iter=400,
        random_state=42,
    )
    knn_snmf = KNeighborsClassifier(n_neighbors=3, n_jobs=-1)

    t0 = time.perf_counter()
    selector_snmf.fit(X_train, y_train)
    X_train_screened = selector_snmf.transform(X_train)
    screen_time_snmf = time.perf_counter() - t0

    t1 = time.perf_counter()
    W_train_snmf = nmf_snmf.fit_transform(X_train_screened)
    mf_time_snmf = time.perf_counter() - t1

    t2 = time.perf_counter()
    knn_snmf.fit(W_train_snmf, y_train)
    fit_time_snmf = time.perf_counter() - t2

    t3 = time.perf_counter()
    X_test_screened = selector_snmf.transform(X_test)
    W_test_snmf = nmf_snmf.transform(X_test_screened)
    preds_snmf = knn_snmf.predict(W_test_snmf)
    test_time_snmf = time.perf_counter() - t3
    acc_snmf = accuracy_score(y_test, preds_snmf)

    # ---------------------------------------------------------
    # FRAMEWORK 2: Standard Unsupervised NMF (Full Space -> K)
    # ---------------------------------------------------------
    nmf_std = NMF(
        n_components=optimal_K,
        init="nndsvda",
        solver="cd",
        max_iter=400,
        random_state=42,
    )
    knn_std = KNeighborsClassifier(n_neighbors=3, n_jobs=-1)
    screen_time_std = 0.0000

    t0 = time.perf_counter()
    W_train_std = nmf_std.fit_transform(X_train)
    mf_time_std = time.perf_counter() - t0

    t1 = time.perf_counter()
    knn_std.fit(W_train_std, y_train)
    fit_time_std = time.perf_counter() - t1

    t2 = time.perf_counter()
    W_test_std = nmf_std.transform(X_test)
    preds_std = knn_std.predict(W_test_std)
    test_time_std = time.perf_counter() - t2
    acc_std = accuracy_score(y_test, preds_std)

    # ---------------------------------------------------------
    # EXPORT FEATURE RECOVERY IMAGES
    # ---------------------------------------------------------
    generate_feature_recovery_comparison_plots(
        X_train=X_train,
        selector=selector_snmf,
        nmf_std=nmf_std,
        nmf_snmf=nmf_snmf,
        attribute_name=attribute_name,
        n_display=6,
    )

    # ---------------------------------------------------------
    # DISPLAY ISOLATED ATTRIBUTE SUMMARY (DECOUPLED AUDIT)
    # ---------------------------------------------------------
    print(f"\nExecution Profile [{attribute_name}]:")
    print(
        f"{'Pipeline Framework Model':<35} | {'Accuracy':<9} | {'Screen Time':<11} | {'MF Time':<9} | {'Train Time':<10} | {'Test Time':<9}"
    )
    print("-" * 105)
    print(
        f"{'Original KNN (Full d-space)':<35} | {acc_knn*100:>6.2f}%   | {screen_time_knn:>8.4f}s  | {mf_time_knn:>6.4f}s  | {fit_time_knn:>7.4f}s   | {test_time_knn:>6.4f}s"
    )
    print(
        f"{f'Standard NMF (Full d-space -> K={optimal_K})':<35} | {acc_std*100:>6.2f}%   | {screen_time_std:>8.4f}s  | {mf_time_std:>6.4f}s  | {fit_time_std:>7.4f}s   | {test_time_std:>6.4f}s"
    )
    print(
        f"{f'Supervised NMF (Strict {optimal_P}% -> K={optimal_K})':<35} | {acc_snmf*100:>6.2f}%   | {screen_time_snmf:>8.4f}s  | {mf_time_snmf:>6.4f}s  | {fit_time_snmf:>7.4f}s   | {test_time_snmf:>6.4f}s"
    )
    print("=" * 105)

    return (
        [acc_knn, acc_std, acc_snmf],
        [screen_time_knn, screen_time_std, screen_time_snmf],
        [mf_time_knn, mf_time_std, mf_time_snmf],
        [fit_time_knn, fit_time_std, fit_time_snmf],
        [test_time_knn, test_time_std, test_time_snmf],
        optimal_P,
        optimal_K,
    )


# ==========================================
# 4. RUNTIME EXECUTION ENTRYPOINT
# ==========================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dir",
        type=str,
        default="D:\\Data\\img_align_celeba",
        help="Absolute path to raw JPEG directory",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=5000,
        help="Total image slice sample count N",
    )
    args = parser.parse_args()

    # Exclusively evaluate Smiling and Wearing_Necktie
    target_attributes = ["Smiling", "Wearing_Necktie"]
    global_results = {}

    for attr in target_attributes:
        X, y = load_celeba_task(
            data_dir=args.dir, attribute_name=attr, N_samples=args.samples
        )
        (
            accs,
            screen_times,
            mf_times,
            fit_times,
            test_times,
            chosen_P,
            chosen_K,
        ) = run_evaluation_showdown(X, y, attribute_name=attr)
        global_results[attr] = {
            "acc": accs,
            "screen": screen_times,
            "mf": mf_times,
            "fit": fit_times,
            "test": test_times,
            "P": chosen_P,
            "K": chosen_K,
        }

    print("\n\n" + "#" * 105)
    print("GLOBAL DISSERTATION METRIC MATRIX COMPILATION")
    print("#" * 105)
    for attr, metrics in global_results.items():
        print(
            f"\nTarget Attribute: {attr} (Locked P={metrics['P']}%, Locked K={metrics['K']})"
        )
        print(
            f" -> Original KNN       | Acc: {metrics['acc'][0]*100:6.2f}% | Screen: {metrics['screen'][0]:.4f}s | MF: {metrics['mf'][0]:.4f}s | Train: {metrics['fit'][0]:.4f}s | Test: {metrics['test'][0]:.4f}s"
        )
        print(
            f" -> Standard NMF       | Acc: {metrics['acc'][1]*100:6.2f}% | Screen: {metrics['screen'][1]:.4f}s | MF: {metrics['mf'][1]:.4f}s | Train: {metrics['fit'][1]:.4f}s | Test: {metrics['test'][1]:.4f}s"
        )
        print(
            f" -> Supervised NMF     | Acc: {metrics['acc'][2]*100:6.2f}% | Screen: {metrics['screen'][2]:.4f}s | MF: {metrics['mf'][2]:.4f}s | Train: {metrics['fit'][2]:.4f}s | Test: {metrics['test'][2]:.4f}s"
        )
    print("#" * 105)