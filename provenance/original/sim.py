import time
import warnings
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import NMF, PCA
from sklearn.feature_selection import SelectPercentile, f_classif
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

# ==============================================================================
# 1. OCCLUDED ADDITIVE PARTS DATASET GENERATOR
# ==============================================================================
def generate_occluded_nmf_dataset(N=6000, d=25000, signal_ratio=0.01, random_state=42):
    """
    Generates a non-negative dataset with severe spatial occlusion:
    - 1% informative additive target features with small Gaussian errors.
    - 99% high-variance background distractor features.
    """
    print(f"Generating Synthetic Occluded Topology (N={N:,}, d={d:,})...")
    np.random.seed(random_state)
    
    # 1. Balanced binary class labels
    y = np.zeros(N, dtype=np.int32)
    y[N // 2:] = 1
    
    d_signal = int(d * signal_ratio)
    d_distractor = d - d_signal
    
    # 2. Additive target signal features (small additive Gaussian error, strong class shift)
    mu_base = 1.0
    delta = 2.0
    sigma_signal = 0.8  # Small additive Gaussian noise
    class_shift = (y * delta)[:, np.newaxis]
    X_signal = mu_base + class_shift + np.random.normal(0, sigma_signal, (N, d_signal))
    
    # 3. High-variance ambient distractors (identical means across classes)
    # Severe ambient variance (sigma=9.0) hijacks unsupervised NMF's Frobenius norm
    mu_distractor = 3.0
    sigma_distractor = 9.0
    X_distractor = np.random.normal(mu_distractor, sigma_distractor, (N, d_distractor))
    
    # 4. Concatenate and clip strictly to the non-negative orthant R_+^d
    X = np.hstack([X_signal, X_distractor])
    X = np.clip(X, a_min=0.0, a_max=None).astype(np.float32)
    
    print(f" -> Signal Subspace     : {d_signal:,} features (1%) | Sigma = {sigma_signal}")
    print(f" -> Distractor Subspace : {d_distractor:,} features (99%) | Sigma = {sigma_distractor}")
    return X, y, d_signal

# ==============================================================================
# 2. DISSERTATION VISUALIZATION ENGINE (3 MECHANISTIC PROOFS)
# ==============================================================================
def generate_dissertation_figures(X, y, W_std, W_sup, accs, screen_times, mf_times, 
                                  train_times, test_times, selector, d_signal, 
                                  optimal_P, optimal_K):
    """
    Generates 3 publication-ready defense figures visualizing data structure,
    latent space separability, and strictly decoupled benchmark timings.
    """
    print("\n[Visualizer] Rendering publication-ready mechanistic figures...")
    
    # --------------------------------------------------------------------------
    # FIGURE 1: Data Structure Variance vs. ANOVA F-Statistic Screening
    # --------------------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    fig.suptitle("Data Structure Profile & Label-Aware Sure Screening (d = 25,000)", 
                 fontsize=14, fontweight='bold')
    
    # Left: Empirical Feature Variance Profile
    variances = np.var(X, axis=0)
    ax1.plot(range(len(variances)), variances, color='#c0392b', linewidth=0.7, label='Empirical Variance s_j^2')
    ax1.axvspan(0, d_signal, color='#27ae60', alpha=0.35, label='True Additive Signal Subspace (1%)')
    ax1.set_title('Empirical Feature Variance Across Dimensions', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Feature Index j', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Variance', fontsize=11, fontweight='bold')
    ax1.grid(True, linestyle=':', alpha=0.6)
    ax1.legend(loc='upper right')
    
    # Right: ANOVA F-Statistic Spectrum
    scores = np.nan_to_num(selector.scores_, 0)
    ax2.plot(range(len(scores)), scores, color='#2c3e50', linewidth=0.7, label='ANOVA F-Statistic (T_j^2)')
    ax2.axvspan(0, d_signal, color='#27ae60', alpha=0.35, label='True Additive Signal Subspace (1%)')
    ax2.set_title('Label-Aware Sure Screening Recovery Profile', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Feature Index j', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Test Statistic', fontsize=11, fontweight='bold')
    ax2.grid(True, linestyle=':', alpha=0.6)
    ax2.legend(loc='upper right')
    
    plt.tight_layout()
    plt.savefig("sNMF_Fig1_Data_Structure_and_Screening.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(" -> Saved chart: 'sNMF_Fig1_Data_Structure_and_Screening.png'")
    
    # --------------------------------------------------------------------------
    # FIGURE 2: Why Supervised NMF Wins (Latent Coordinate Space Separability)
    # --------------------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Why Supervised NMF Outperforms: Latent Space Separability Comparison", 
                 fontsize=14, fontweight='bold')
    
    # Subplot 1: Ambient d-space (via 2D PCA projection for visualization)
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X[:1000])  # Plot first 1,000 samples for visual clarity
    y_sub = y[:1000]
    
    axes[0].scatter(X_pca[y_sub==0, 0], X_pca[y_sub==0, 1], alpha=0.6, s=15, color='#3498db', label='Class 0')
    axes[0].scatter(X_pca[y_sub==1, 0], X_pca[y_sub==1, 1], alpha=0.6, s=15, color='#e67e22', label='Class 1')
    axes[0].set_title('1. Full d-Space (PCA Projection)\nSevere Occlusion & Class Overlap', fontsize=11, fontweight='bold')
    axes[0].set_xlabel('Component 1', fontsize=10, fontweight='bold')
    axes[0].set_ylabel('Component 2', fontsize=10, fontweight='bold')
    axes[0].grid(True, linestyle=':', alpha=0.6)
    axes[0].legend(loc='upper right')
    
    # Subplot 2: Standard Unsupervised NMF Latent Space (W_std)
    w_std_sub = W_std[:1000, :2]
    axes[1].scatter(w_std_sub[y_sub==0, 0], w_std_sub[y_sub==0, 1], alpha=0.6, s=15, color='#3498db', label='Class 0')
    axes[1].scatter(w_std_sub[y_sub==1, 0], w_std_sub[y_sub==1, 1], alpha=0.6, s=15, color='#e67e22', label='Class 1')
    axes[1].set_title('2. Standard Unsupervised NMF Latent Space\nHijacked by Background Distractor Variance', fontsize=11, fontweight='bold')
    axes[1].set_xlabel('Latent Coordinate w_1', fontsize=10, fontweight='bold')
    axes[1].set_ylabel('Latent Coordinate w_2', fontsize=10, fontweight='bold')
    axes[1].grid(True, linestyle=':', alpha=0.6)
    axes[1].legend(loc='upper right')
    
    # Subplot 3: Supervised NMF Latent Space (W_sup)
    w_sup_sub = W_sup[:1000, :2]
    axes[2].scatter(w_sup_sub[y_sub==0, 0], w_sup_sub[y_sub==0, 1], alpha=0.7, s=15, color='#3498db', label='Class 0')
    axes[2].scatter(w_sup_sub[y_sub==1, 0], w_sup_sub[y_sub==1, 1], alpha=0.7, s=15, color='#e67e22', label='Class 1')
    axes[2].set_title('3. Supervised NMF Latent Space (sNMF)\nLinear Separability Rescued', fontsize=11, fontweight='bold')
    axes[2].set_xlabel('Latent Coordinate w_1', fontsize=10, fontweight='bold')
    axes[2].set_ylabel('Latent Coordinate w_2', fontsize=10, fontweight='bold')
    axes[2].grid(True, linestyle=':', alpha=0.6)
    axes[2].legend(loc='upper right')
    
    plt.tight_layout()
    plt.savefig("sNMF_Fig2_Latent_Space_Separability.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(" -> Saved chart: 'sNMF_Fig2_Latent_Space_Separability.png'")

    # --------------------------------------------------------------------------
    # FIGURE 3: Classification Accuracy & Fully Decoupled Runtime Breakdown
    # --------------------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5.5))
    fig.suptitle(f"Benchmark Performance & Fully Decoupled Runtimes (P = {optimal_P}%, K = {optimal_K})", 
                 fontsize=14, fontweight='bold')
    
    models = ['Raw KNN\n(Full d-space)', 'Standard NMF\n(Full d-space)', 'Supervised NMF\n(Filtered Subspace)']
    colors_acc = ['#7f8c8d', '#c0392b', '#27ae60']
    
    # Left: Out-of-Sample Classification Accuracy
    bars1 = ax1.bar(models, [a * 100 for a in accs], color=colors_acc, width=0.5, edgecolor='black', alpha=0.9)
    ax1.set_ylabel('Test Accuracy (%)', fontsize=12, fontweight='bold')
    ax1.set_ylim(0, 110)
    ax1.axhline(50.0, color='gray', linestyle='--', linewidth=1.0, label='Random Guessing (50%)')
    ax1.set_title('Out-of-Sample Classification Accuracy', fontsize=12, fontweight='bold')
    ax1.grid(axis='y', linestyle=':', alpha=0.6)
    ax1.legend(loc='upper left')
    
    for bar in bars1:
        height = bar.get_height()
        ax1.annotate(f"{height:.2f}%",
                     xy=(bar.get_x() + bar.get_width() / 2, height),
                     xytext=(0, 5),  
                     textcoords="offset points",
                     ha='center', va='bottom', fontsize=11, fontweight='bold')
        
    # Right: Grouped Bar Chart of Fully Decoupled Execution Times
    x = np.arange(len(models))
    width = 0.20
    
    ax2.bar(x - 1.5*width, screen_times, width, label='Screen Time (ANOVA F-Test)', color='#8e44ad', edgecolor='black')
    ax2.bar(x - 0.5*width, mf_times, width, label='MF Time (Factorization)', color='#e74c3c', edgecolor='black')
    ax2.bar(x + 0.5*width, train_times, width, label='Train Time (Classifier Fit Only)', color='#f39c12', edgecolor='black')
    ax2.bar(x + 1.5*width, test_times, width, label='Test Time (Inference)', color='#3498db', edgecolor='black')
    
    ax2.set_ylabel('Execution Time (Seconds)', fontsize=12, fontweight='bold')
    ax2.set_title('Strictly Decoupled Wall-Clock Execution Times', fontsize=12, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(models, fontweight='bold')
    ax2.grid(axis='y', linestyle=':', alpha=0.6)
    ax2.legend(loc='upper right')
    
    plt.tight_layout()
    plt.savefig("sNMF_Fig3_Performance_and_Timing.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(" -> Saved chart: 'sNMF_Fig3_Performance_and_Timing.png'")

# ==============================================================================
# 3. PROFILED BENCHMARK & SHOWDOWN ENGINE
# ==============================================================================
def run_benchmark_showdown(X, y, d_signal):
    N_samples, d_dim = X.shape
    print("\n" + "="*125)
    print(f"RUNNING BENCHMARK AUDIT (N={N_samples:,}, d={d_dim:,} | cv=5)")
    print("="*125)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # --------------------------------------------------------------------------
    # PIPELINE 1: Raw KNN (Full d-space Euclidean Distance)
    # --------------------------------------------------------------------------
    print("Executing Raw KNN baseline across full Euclidean feature space...")
    m1_knn = KNeighborsClassifier(n_neighbors=5, n_jobs=-1)
    m1_screen_time = 0.0000  # Zero screening time
    m1_mf_time = 0.0000      # Zero factorization time
    
    t0 = time.time()
    m1_knn.fit(X_train, y_train)
    m1_train_time = time.time() - t0  # STRICTLY classifier fit time
    
    t1 = time.time()
    m1_preds = m1_knn.predict(X_test)
    m1_test_time = time.time() - t1   # Out-of-sample inference time
    m1_total_time = m1_screen_time + m1_mf_time + m1_train_time + m1_test_time
    m1_acc = accuracy_score(y_test, m1_preds)
    
    # --------------------------------------------------------------------------
    # OFFLINE TUNING PHASE: Streamlined 90-Point 5-Fold Grid Search
    # --------------------------------------------------------------------------
    print("\nRunning offline parameter co-optimization (10 Percentiles x 9 Latent Ranks | cv=5)...")
    tuning_pipeline = Pipeline([
        ('selector', SelectPercentile(score_func=f_classif)),
        ('nmf', NMF(init='nndsvda', solver='cd', max_iter=300, random_state=42)),
        ('knn', KNeighborsClassifier(n_neighbors=5, n_jobs=-1))
    ])
    
    param_grid = {
        'selector__percentile': [0.1, 0.25, 0.5, 1.0, 1.5, 2.0, 5.0, 10.0, 20.0, 30.0],
        'nmf__n_components': [2, 3, 4, 5, 8, 10, 15, 20, 30]
    }
    
    grid_search = GridSearchCV(tuning_pipeline, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    
    optimal_P = grid_search.best_params_['selector__percentile']
    optimal_K = grid_search.best_params_['nmf__n_components']
    print(f" -> Optimal Parameters Locked: Percentile P = {optimal_P}%, Rank K = {optimal_K}")
    
    # --------------------------------------------------------------------------
    # PIPELINE 3: Supervised NMF (Dissertation Algorithm: Pre-Filter -> NMF -> KNN)
    # --------------------------------------------------------------------------
    print(f"Executing Supervised NMF (Strict {optimal_P}% Screening -> K={optimal_K})...")
    m3_selector = SelectPercentile(score_func=f_classif, percentile=optimal_P)
    m3_nmf = NMF(n_components=optimal_K, init='nndsvda', solver='cd', max_iter=400, random_state=42)
    m3_knn = KNeighborsClassifier(n_neighbors=5, n_jobs=-1)
    
    # Phase 1: STRICTLY Feature Screening Time (Screen Time)
    t0 = time.time()
    X_train_filtered = m3_selector.fit_transform(X_train, y_train)
    m3_screen_time = time.time() - t0
    
    # Phase 2: STRICTLY Matrix Factorization Time (MF Time)
    t1 = time.time()
    W3_train = m3_nmf.fit_transform(X_train_filtered)
    m3_mf_time = time.time() - t1
    
    # Phase 3: STRICTLY Downstream Classifier Fit Time (Train Time)
    t2 = time.time()
    m3_knn.fit(W3_train, y_train)
    m3_train_time = time.time() - t2
    
    # Phase 4: STRICTLY Out-of-Sample Inference Time (Test Time)
    t3 = time.time()
    X_test_filtered = m3_selector.transform(X_test)
    W3_test = m3_nmf.transform(X_test_filtered)
    m3_preds = m3_knn.predict(W3_test)
    m3_test_time = time.time() - t3
    
    m3_total_time = m3_screen_time + m3_mf_time + m3_train_time + m3_test_time
    m3_acc = accuracy_score(y_test, m3_preds)
    
    # --------------------------------------------------------------------------
    # PIPELINE 2: Standard Unsupervised NMF (Full d-space -> K*)
    # --------------------------------------------------------------------------
    print(f"Executing Standard Unsupervised NMF across full feature space (K={optimal_K})...")
    m2_nmf = NMF(n_components=optimal_K, init='nndsvda', solver='cd', max_iter=400, random_state=42)
    m2_knn = KNeighborsClassifier(n_neighbors=5, n_jobs=-1)
    m2_screen_time = 0.0000  # Zero screening time
    
    # Phase 1: STRICTLY Full d-space Matrix Factorization (MF Time)
    t0 = time.time()
    W2_train = m2_nmf.fit_transform(X_train)
    m2_mf_time = time.time() - t0
    
    # Phase 2: STRICTLY Downstream Classifier Fit Time (Train Time)
    t1 = time.time()
    m2_knn.fit(W2_train, y_train)
    m2_train_time = time.time() - t1
    
    # Phase 3: STRICTLY Out-of-Sample Inference Time (Test Time)
    t2 = time.time()
    W2_test = m2_nmf.transform(X_test)
    m2_preds = m2_knn.predict(W2_test)
    m2_test_time = time.time() - t2
    
    m2_total_time = m2_screen_time + m2_mf_time + m2_train_time + m2_test_time
    m2_acc = accuracy_score(y_test, m2_preds)

    # --------------------------------------------------------------------------
    # DISPLAY CONSOLE METRIC MATRIX (STRICTLY DECOUPLED TIMINGS)
    # --------------------------------------------------------------------------
    print(f"\nExecution Profile [Synthetic Occluded Topology: N={N_samples:,}, d={d_dim:,}]:")
    print(f"{'Algorithm Pipeline':<33} | {'Accuracy':<9} | {'Screen Time':<11} | {'MF Time':<10} | {'Train Time':<10} | {'Test Time':<10} | {'Total Time':<10}")
    print("-" * 125)
    print(f"{'Raw KNN (Full d-space)':<33} | {m1_acc*100:>6.2f}%   | {m1_screen_time:>8.3f}s   | {m1_mf_time:>7.3f}s   | {m1_train_time:>7.3f}s   | {m1_test_time:>7.3f}s   | {m1_total_time:>7.3f}s")
    print(f"{f'Standard NMF (K={optimal_K})':<33} | {m2_acc*100:>6.2f}%   | {m2_screen_time:>8.3f}s   | {m2_mf_time:>7.3f}s   | {m2_train_time:>7.3f}s   | {m2_test_time:>7.3f}s   | {m2_total_time:>7.3f}s")
    print(f"{f'Supervised NMF ({optimal_P}% -> K={optimal_K})':<33} | {m3_acc*100:>6.2f}%   | {m3_screen_time:>8.3f}s   | {m3_mf_time:>7.3f}s   | {m3_train_time:>7.3f}s   | {m3_test_time:>7.3f}s   | {m3_total_time:>7.3f}s")
    print("=" * 125)
    
    # Render mechanistic figures
    generate_dissertation_figures(
        X_train, y_train, W2_train, W3_train,
        [m1_acc, m2_acc, m3_acc],
        [m1_screen_time, m2_screen_time, m3_screen_time],
        [m1_mf_time, m2_mf_time, m3_mf_time],
        [m1_train_time, m2_train_time, m3_train_time],
        [m1_test_time, m2_test_time, m3_test_time],
        m3_selector,
        d_signal,
        optimal_P,
        optimal_K
    )

# ==============================================================================
# 4. RUNTIME ENTRYPOINT
# ==============================================================================
if __name__ == "__main__":
    X_synth, y_synth, d_sig = generate_occluded_nmf_dataset(
        N=6000, d=25000, signal_ratio=0.01, random_state=42
    )
    run_benchmark_showdown(X_synth, y_synth, d_sig)