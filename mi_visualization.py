import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import altair as alt

def create_log_binned_histogram_linear(ngram_dfs, titles, n_bins=10, figsize=(24, 8), frequency_column='frequency'):
    """
    Create a row of log-binned histograms with linear y-axis for different n-gram datasets
    with consistent scales for comparison.
    
    Parameters:
    -----------
    ngram_dfs : list of pandas.DataFrame
        List of DataFrames containing n-grams with a specified frequency column
    titles : list of str
        Titles for each histogram
    n_bins : int, optional
        Number of logarithmic bins to use
    figsize : tuple, optional
        Figure size for the plot (width, height)
    frequency_column : str, optional
        Name of the column containing frequency data
    
    Returns:
    --------
    matplotlib.figure.Figure
        The figure containing the row of histograms
    """
    n_cols = len(ngram_dfs)
    fig, axes = plt.subplots(1, n_cols, figsize=figsize)
    
    # If there's only one column, make sure axes is an array
    if n_cols == 1:
        axes = np.array([axes])
    
    # Find global min and max for consistent binning
    global_min_freq = min(df[frequency_column].min() for df in ngram_dfs)
    global_max_freq = max(df[frequency_column].max() for df in ngram_dfs)
    
    # Create a single set of log-spaced bin edges for all histograms
    global_log_bins = np.logspace(np.log10(global_min_freq), np.log10(global_max_freq), n_bins + 1)
    
    # Calculate all histograms first to determine global y-axis range
    all_counts = []
    for df in ngram_dfs:
        counts, _ = np.histogram(df[frequency_column], bins=global_log_bins)
        all_counts.append(counts)
    
    # Determine global y-axis limits for linear scale
    global_max_count = max(count.max() for count in all_counts)
    
    # Create custom x-axis labels (consistent for all plots)
    step = max(1, len(global_log_bins) // 10)  # Show ~10 labels max
    tick_positions = range(0, n_bins, step)
    tick_labels = [f'{int(global_log_bins[i])}-{int(global_log_bins[i+1])}' for i in range(0, len(global_log_bins)-1, step)]
    
    for i, (df, title, counts) in enumerate(zip(ngram_dfs, titles, all_counts)):
        # LINEAR SCALE PLOT
        axes[i].bar(range(len(counts)), counts, width=0.8, alpha=0.7, color='lightgreen', edgecolor='black')
        axes[i].set_xticks(tick_positions)
        axes[i].set_xticklabels(tick_labels, rotation=90, ha='center')
        axes[i].set_xlabel('Freq. of n-gram\noccurrence')
        axes[i].set_ylabel('Number of N-grams')
        axes[i].set_title(title)
        axes[i].grid(True, alpha=0.3, axis='y')
        
        # Set aspect ratio to make height/width ~ 2
        axes[i].set_box_aspect(2)
        
        # Set consistent y-limits for linear plots
        axes[i].set_ylim(0, global_max_count * 1.1)  # Add 10% buffer at the top
    
    plt.tight_layout()
    return fig

def create_log_binned_histogram_normalized(ngram_dfs, titles, n_bins=10, figsize=(24, 8), frequency_column='frequency'):
    """
    Create a row of log-binned histograms with logarithmic y-axis
    for different n-gram datasets with consistent scales for comparison.
    
    Parameters:
    -----------
    ngram_dfs : list of pandas.DataFrame
        List of DataFrames containing n-grams with a specified frequency column
    titles : list of str
        Titles for each histogram
    n_bins : int, optional
        Number of logarithmic bins to use
    figsize : tuple, optional
        Figure size for the plot (width, height)
    frequency_column : str, optional
        Name of the column containing frequency data
    
    Returns:
    --------
    matplotlib.figure.Figure
        The figure containing the row of histograms
    """
    n_cols = len(ngram_dfs)
    fig, axes = plt.subplots(1, n_cols, figsize=figsize)
    
    # If there's only one column, make sure axes is an array
    if n_cols == 1:
        axes = np.array([axes])
    
    # Find global min and max for consistent binning
    global_min_freq = min(df[frequency_column].min() for df in ngram_dfs)
    global_max_freq = max(df[frequency_column].max() for df in ngram_dfs)
    
    # Create a single set of log-spaced bin edges for all histograms
    global_log_bins = np.logspace(np.log10(global_min_freq), np.log10(global_max_freq), n_bins + 1)
    
    # Calculate all histograms first to determine global y-axis range
    all_counts = []
    for df in ngram_dfs:
        counts, _ = np.histogram(df[frequency_column], bins=global_log_bins)
        all_counts.append(counts)
    
    # Determine global y-axis limits for log scale
    global_max_count = max(count.max() for count in all_counts)
    
    # Create custom x-axis labels (consistent for all plots)
    step = max(1, len(global_log_bins) // 10)  # Show ~10 labels max
    tick_positions = range(0, n_bins, step)
    tick_labels = [f'{int(global_log_bins[i])}-{int(global_log_bins[i+1])}' for i in range(0, len(global_log_bins)-1, step)]
    
    for i, (df, title, counts) in enumerate(zip(ngram_dfs, titles, all_counts)):
        # LOG SCALE PLOT
        axes[i].bar(range(len(counts)), counts, width=0.8, alpha=0.7, color='skyblue', edgecolor='black')
        axes[i].set_yscale('log')  # Use logarithmic y-axis
        axes[i].set_xticks(tick_positions)
        axes[i].set_xticklabels(tick_labels, rotation=90, ha='center')
        axes[i].set_xlabel('Frequency Range')
        axes[i].set_ylabel('Number of N-grams (log scale)')
        axes[i].set_title(title)
        axes[i].grid(True, alpha=0.3, axis='y')
        
        # Set aspect ratio to make height/width ~ 2
        axes[i].set_box_aspect(2)
        
        # Set consistent y-limits for log plots (1 to slightly above max count)
        max_count_with_buffer = max(global_max_count * 1.1, 2)  # Ensure at least 2 for log scale
        axes[i].set_ylim(0.9, max_count_with_buffer)  # Start slightly below 1 for log scale
    
    plt.tight_layout()
    return fig





# ------------------------------------------------------------
# Parametric Zipf-like scatter (rank on x, chosen metric on y)
# ------------------------------------------------------------
def create_zipf_plot_parametric(
    session_ngrams,
    *,
    y_col: str = 'normalized_frequency',
    color_column: str = 'ngram_length',
    color_mode: str = None,       # 'categorical' | 'continuous' | None (auto)
    rank_within: str = 'global',  # 'global' or 'length'
    normalize_frequency: bool = True,
    x_log: bool = True,
    y_log: str = 'auto',          # True | False | 'auto'
    width: int = 700,
    height: int = 500,
    color_scheme: str = None,
    color_range: list = None,
    color_title: str = None
):
    """
    Create a rank-vs-metric scatter with flexible y-axis and color palette.

    Parameters
    ----------
    session_ngrams : pd.DataFrame
        Must contain at least 'ngram' and 'frequency'.
    y_col : str
        Column to plot on y-axis (e.g., 'normalized_frequency', 'mi_in_session').
    color_column : str
        Column used for color encoding (categorical or numeric).
    rank_within : {'global','length'}
        If 'global', compute a single rank across all rows by frequency.
        If 'length', compute rank separately per ngram_length.
    normalize_frequency : bool
        If True and y_col == 'normalized_frequency', compute it if missing.
    x_log : bool
        Log scale for x (rank). Typically True for Zipf-like plots.
    y_log : bool | 'auto'
        Log scale for y. If 'auto', uses log only if all y > 0.
    color_mode : Optional[str]
        Force color encoding type regardless of dtype: 'categorical' (N) or 'continuous' (Q).
        If None, the type is inferred from the column dtype.
    color_scheme : Optional[str]
        Altair categorical/continuous color scheme name (e.g., 'tableau10', 'category20', 'viridis').
    color_range : Optional[list]
        Custom list of colors (e.g., ['#1f77b4', '#ff7f0e']).
    color_title : Optional[str]
        Override color legend title.
    """
    import pandas as pd
    import numpy as np
    import altair as alt

    df = session_ngrams.copy()
    if 'frequency' not in df.columns:
        raise ValueError("session_ngrams must include 'frequency' column")

    # Optionally create normalized frequency
    if y_col == 'normalized_frequency' and normalize_frequency:
        if 'normalized_frequency' not in df.columns:
            max_freq = df['frequency'].max() or 1
            df['normalized_frequency'] = df['frequency'] / max_freq

    # Compute rank (global or within length)
    if rank_within == 'length':
        if 'ngram_length' not in df.columns:
            raise ValueError("'ngram_length' is required for rank_within='length'")
        df = df.sort_values(['ngram_length', 'frequency'], ascending=[True, False])
        df['rank'] = df.groupby('ngram_length').cumcount() + 1
    else:
        df = df.sort_values('frequency', ascending=False)
        df['rank'] = np.arange(1, len(df) + 1)

    # Clean y values
    if y_col not in df.columns:
        raise ValueError(f"Column '{y_col}' not found in DataFrame.")
    plot_df = df[['ngram', y_col, color_column, 'rank']].dropna()

    # Decide y scale
    use_y_log = False
    if y_log is True:
        use_y_log = True
    elif y_log == 'auto':
        if len(plot_df) and (plot_df[y_col] > 0).all():
            use_y_log = True

    x_scale = alt.Scale(type='log') if x_log else alt.Undefined
    y_scale = alt.Scale(type='log') if use_y_log else alt.Undefined

    # Choose color encoding type
    # Determine encoding type for color
    if color_mode == 'categorical':
        enc_type = 'N'
    elif color_mode == 'continuous':
        enc_type = 'Q'
    else:
        color_is_numeric = pd.api.types.is_numeric_dtype(plot_df[color_column])
        enc_type = 'Q' if color_is_numeric else 'N'
    legend_title = color_title or color_column.replace('_', ' ').title()

    scale_kwargs = {}
    if color_scheme:
        scale_kwargs['scheme'] = color_scheme
    if color_range:
        scale_kwargs['range'] = color_range

    if scale_kwargs:
        color_enc = alt.Color(f'{color_column}:{enc_type}', title=legend_title, scale=alt.Scale(**scale_kwargs))
    else:
        color_enc = alt.Color(f'{color_column}:{enc_type}', title=legend_title)

    scatter = alt.Chart(plot_df).mark_circle(size=40, opacity=0.6).encode(
        x=alt.X('rank:Q', title='Rank (log scale)' if x_log else 'Rank', scale=x_scale),
        y=alt.Y(f'{y_col}:Q', title=y_col.replace('_', ' ').title(), scale=y_scale),
        color=color_enc,
        tooltip=[
            alt.Tooltip('ngram:N', title='N-gram'),
            alt.Tooltip('rank:Q', title='Rank', format=',d'),
            alt.Tooltip('frequency:Q', title='Frequency', format=',d') if 'frequency' in df.columns else alt.Tooltip('rank:Q', title='Rank')
        ]
    ).properties(width=width, height=height)

    return scatter


# ------------------------------------------------------------
# MI histograms by n-gram length (raw and normalized)
# ------------------------------------------------------------
def plot_mi_histogram_by_length(
    df,
    *,
    length_col: str = 'ngram_length',
    lengths=(2, 3),
    value_col: str = 'mi_in_session',
    bins: int = 30,
    density: bool = True,
    alpha: float = 0.6,
    figsize=(7, 5),
    title: str = 'Raw MI distributions',
    xlabel: str = 'Mutual Information',
    ylabel: str = 'Density'
):
    """
    Plot a single histogram panel of MI values by n-gram length.
    Returns (fig, ax).
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=figsize)
    for length in lengths:
        subset = df[df[length_col] == length]
        if value_col not in subset.columns:
            continue
        ax.hist(
            subset[value_col].dropna(),
            bins=bins,
            alpha=alpha,
            label=f'{length}-grams',
            density=density
        )
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    return fig, ax


def plot_mi_histogram_z_by_length(
    df,
    *,
    length_col: str = 'ngram_length',
    lengths=(2, 3),
    mi_col: str = 'mi_in_session',
    mi_z_col: str = None,
    bins: int = 30,
    density: bool = True,
    alpha: float = 0.6,
    figsize=(7, 5),
    title: str = 'Normalized MI (z-scores)',
    xlabel: str = 'MI z-score',
    ylabel: str = 'Density',
    show_guides: bool = True
):
    """
    Plot a single histogram panel of MI z-scores by n-gram length.
    Returns (fig, ax).
    """
    import matplotlib.pyplot as plt

    z_col = mi_z_col if mi_z_col is not None else f'{mi_col}_z'
    fig, ax = plt.subplots(figsize=figsize)
    for length in lengths:
        subset = df[df[length_col] == length]
        if z_col not in subset.columns:
            continue
        ax.hist(
            subset[z_col].dropna(),
            bins=bins,
            alpha=alpha,
            label=f'{length}-grams',
            density=density
        )
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    if show_guides:
        ax.axvline(0, color='black', linestyle='--', alpha=0.5, label='mean')
        ax.axvline(2, color='red', linestyle='--', alpha=0.3, label='z=2')
        ax.axvline(-2, color='red', linestyle='--', alpha=0.3, label='z=-2')
    return fig, ax


def plot_mi_histograms_before_after(
    df,
    *,
    length_col: str = 'ngram_length',
    lengths=(2, 3),
    mi_col: str = 'mi_in_session',
    mi_z_col: str = None,
    bins: int = 30,
    density: bool = True,
    alpha: float = 0.6,
    figsize=(14, 5)
):
    """
    Plot side-by-side histograms: raw MI (left) and normalized MI z (right), by n-gram length.
    Returns (fig, (ax_raw, ax_norm)).
    """
    import matplotlib.pyplot as plt

    z_col = mi_z_col if mi_z_col is not None else f'{mi_col}_z'
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Left panel: raw MI
    for length in lengths:
        subset = df[df[length_col] == length]
        if mi_col not in subset.columns:
            continue
        axes[0].hist(
            subset[mi_col].dropna(),
            bins=bins,
            alpha=alpha,
            label=f'{length}-grams',
            density=density
        )
    axes[0].set_xlabel('Mutual Information')
    axes[0].set_ylabel('Density')
    axes[0].set_title('Before: Raw MI distributions')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Right panel: normalized MI
    for length in lengths:
        subset = df[df[length_col] == length]
        if z_col not in subset.columns:
            continue
        axes[1].hist(
            subset[z_col].dropna(),
            bins=bins,
            alpha=alpha,
            label=f'{length}-grams',
            density=density
        )
    axes[1].set_xlabel('MI z-score')
    axes[1].set_ylabel('Density')
    axes[1].set_title('After: Normalized MI (z-scores)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[1].axvline(0, color='black', linestyle='--', alpha=0.5, label='mean')
    axes[1].axvline(2, color='red', linestyle='--', alpha=0.3, label='z=2')
    axes[1].axvline(-2, color='red', linestyle='--', alpha=0.3, label='z=-2')

    return fig, axes




# ------------------------------------------------------------
# Timeline visualization for phrase usage (simplified)
# ------------------------------------------------------------
import re
import os


def plot_phrase_timeline(
    df_turns,
    phrases,
    *,
    session_col: str = 'session',
    speaker_col: str = 'speaker',
    text_col: str = 'text',
    index_col: str = 'session_index',
    global_index: bool = True,
    case_sensitive: bool = False,
    show_session_boundaries: bool = False,
    ordered_speakers=None,
    color_scheme: str = 'tableau20',
    altair_data_dir: str = 'altair',
    save_html: str = None,
):
    """
    Plot a simple n-gram usage timeline.

    - x-axis: timeline index (global across sessions if global_index=True, otherwise per-session index)
    - y-axis: speakers
    - color: phrase
    - dots: occurrences of phrases

    Parameters
    ----------
    df_turns : pd.DataFrame
        Turn-level dataframe containing at least session, speaker, text, and index columns.
    phrases : List[str]
        Phrases (n-grams) to visualize.
    session_col, speaker_col, text_col, index_col : str
        Column names in df_turns.
    global_index : bool
        If True, create a monotonically increasing index across sessions for x-axis.
    case_sensitive : bool
        If True, phrase matching is case-sensitive. Default False.
    show_session_boundaries : bool
        If True, draw vertical rules at session boundaries.
    ordered_speakers : Optional[List[str]]
        If provided, y-axis will use this order; otherwise speakers are sorted alphabetically.
    color_scheme : str
        Altair categorical scheme for phrase colors.
    altair_data_dir : str
        Directory where Altair saves data JSON artifacts.
    save_html : Optional[str]
        If provided, save the chart as an HTML file at this path.

    Returns
    -------
    alt.Chart
        The Altair timeline chart.
    """
    import pandas as pd

    if not isinstance(phrases, (list, tuple)) or len(phrases) == 0:
        raise ValueError("phrases must be a non-empty list of strings")

    # Guard and copy
    if df_turns is None or len(df_turns) == 0:
        raise ValueError("df_turns is empty")
    df = df_turns.copy()

    # Ensure required columns exist
    missing = [c for c in [session_col, speaker_col, text_col, index_col] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in df_turns: {missing}")

    # Sort and create timeline index
    df[text_col] = df[text_col].astype(str)
    df = df.sort_values([session_col, index_col]).reset_index(drop=True)
    if global_index:
        df['x'] = range(1, len(df) + 1)
    else:
        df['x'] = df[index_col].astype(int)

    # Build mentions table
    matches_frames = []
    flags = 0 if case_sensitive else re.IGNORECASE

    for phrase in phrases:
        pattern = re.compile(rf"\b{re.escape(phrase)}\b", flags)
        mask = df[text_col].str.contains(pattern, na=False)
        if not mask.any():
            continue
        sub = df.loc[mask, [session_col, speaker_col, text_col, 'x']].copy()
        sub['phrase'] = phrase
        matches_frames.append(sub)

    if not matches_frames:
        # Return an empty chart with a message
        empty_df = pd.DataFrame({
            'x': [], 'speaker': [], 'phrase': []
        })
        chart = alt.Chart(empty_df).mark_text(text='No matches for provided phrases').encode()
        return chart

    mentions = pd.concat(matches_frames, ignore_index=True)

    # Speaker order
    if ordered_speakers is not None and len(ordered_speakers) > 0:
        speaker_sort = ordered_speakers
    else:
        speaker_sort = sorted(mentions[speaker_col].dropna().unique().tolist())

    # Altair data saving directory (Altair v5 uses 'filename' rather than 'data_dir')
    try:
        alt.data_transformers.enable('json', filename=f"{altair_data_dir}/{{prefix}}-{{hash}}.json")
    except Exception:
        try:
            # Backward compatibility for older Altair versions
            alt.data_transformers.enable('json', data_dir=altair_data_dir)
        except Exception:
            pass

    # Main scatter
    tooltip = [
        alt.Tooltip(f'{"phrase"}:N', title='Phrase'),
        alt.Tooltip(f'{speaker_col}:N', title='Speaker'),
        alt.Tooltip(f'{session_col}:N', title='Session'),
        alt.Tooltip('x:Q', title='Index'),
        alt.Tooltip(f'{text_col}:N', title='Text')
    ]

    base = alt.Chart(mentions)
    points = base.mark_circle(size=80, opacity=0.8).encode(
        x=alt.X('x:Q', title='Timeline index'),
        y=alt.Y(f'{speaker_col}:N', sort=speaker_sort, title='Speaker'),
        color=alt.Color('phrase:N', title='Phrase', scale=alt.Scale(scheme=color_scheme)),
        tooltip=tooltip
    )

    layers = [points]

    if show_session_boundaries:
        # Compute first index per session for vertical rules
        boundaries = (
            df.groupby(session_col)['x'].min().reset_index()
        )
        rules = alt.Chart(boundaries).mark_rule(color='gray', strokeDash=[4, 4], opacity=0.5).encode(
            x='x:Q'
        )
        labels = alt.Chart(boundaries).mark_text(align='left', baseline='bottom', dy=-6, angle=270, fontSize=10).encode(
            x='x:Q', text=f'{session_col}:N'
        )
        layers.extend([rules, labels])

    chart = alt.layer(*layers).properties(
        width=900,
        height=max(120, 20 * max(1, len(speaker_sort))),
        title='Phrase usage timeline'
    ).configure_view(stroke=None)

    # Save html if requested
    if save_html:
        out_dir = os.path.dirname(save_html)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        chart.save(save_html)

    return chart


# ------------------------------------------------------------
# Semantic clustering of phrases
# ------------------------------------------------------------
def cluster_phrases_semantic(
    phrases,
    *,
    model_name: str = 'sentence-transformers/all-MiniLM-L6-v2',
    algorithm: str = 'hdbscan',            # 'hdbscan' | 'agglomerative' | 'kmeans' | 'dbscan'
    num_clusters: int = None,              # for 'kmeans' or 'agglomerative' (if distance_threshold is None)
    min_cluster_size: int = 5,             # for 'hdbscan' and as min_samples for 'dbscan'
    distance_threshold: float = None,      # for 'agglomerative' (alternative to num_clusters)
    similarity_metric: str = 'cosine',     # currently used in clustering where supported
    reduce_to_2d: bool = True,
    reducer: str = 'umap',                 # 'umap' | 'pca'
    random_state: int = 42,
    return_embeddings: bool = False
):
    """
    Cluster a list of phrases by semantic similarity.

    Parameters
    ----------
    phrases : List[str]
        Input phrases (strings). Duplicates are allowed; they are handled with stable indexing.
    model_name : str
        Sentence-Transformers model name. Falls back to TF-IDF if unavailable.
    algorithm : str
        Clustering algorithm: 'hdbscan' (preferred), 'agglomerative', 'kmeans', or 'dbscan'.
    num_clusters : Optional[int]
        Number of clusters (used by kmeans; optional for agglomerative if distance_threshold provided).
    min_cluster_size : int
        Minimal cluster size for HDBSCAN; used as min_samples for DBSCAN.
    distance_threshold : Optional[float]
        Agglomerative distance threshold (cosine distance). If set, num_clusters is ignored.
    similarity_metric : str
        Similarity metric; cosine is standard for embeddings.
    reduce_to_2d : bool
        If True, add 2D projection columns (x, y) using UMAP (if installed) or PCA fallback.
    reducer : str
        'umap' or 'pca' for 2D projection.
    random_state : int
        Random seed for kmeans/UMAP/PCA.
    return_embeddings : bool
        If True, include per-phrase embedding vectors in the output DataFrame.

    Returns
    -------
    clusters_df : pd.DataFrame
        Columns: phrase, cluster_id, is_representative, distance_to_centroid, (x,y if projected), (embedding if requested)
    cluster_info : pd.DataFrame
        Columns: cluster_id, size, representative_phrase, centroid (vector), example_phrases (list)
    """
    import numpy as np
    import pandas as pd

    if phrases is None or len(phrases) == 0:
        raise ValueError("phrases must be a non-empty list of strings")

    # Deduplicate while preserving order; keep index mapping
    original_indices = list(range(len(phrases)))
    phrase_series = pd.Series(phrases, dtype=str)
    # Work on unique phrases for efficiency in embedding; map back later
    unique_phrases = phrase_series.drop_duplicates().tolist()

    # Embedding: Sentence-Transformers if available; else TF-IDF fallback
    embeddings_unique = None
    used_encoder = 'st'
    try:
        from sentence_transformers import SentenceTransformer
        st_model = SentenceTransformer(model_name)
        embeddings_unique = st_model.encode(unique_phrases, show_progress_bar=False, normalize_embeddings=True)
        embeddings_unique = np.asarray(embeddings_unique, dtype=np.float32)
    except Exception:
        # Fallback to TF-IDF
        used_encoder = 'tfidf'
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.preprocessing import normalize as sk_normalize
        vec = TfidfVectorizer(ngram_range=(1, 3), min_df=1)
        embeddings_unique = vec.fit_transform(unique_phrases)
        embeddings_unique = sk_normalize(embeddings_unique, norm='l2', copy=False)

    # Build full embeddings aligned with original phrase order
    # Map unique embeddings back to all instances
    unique_index_map = {p: i for i, p in enumerate(unique_phrases)}
    unique_idx_for_all = phrase_series.map(unique_index_map).to_numpy()
    if used_encoder == 'tfidf':
        # Sparse matrix indexing; stack rows for all phrases
        embeddings_all = embeddings_unique[unique_idx_for_all]
    else:
        embeddings_all = embeddings_unique[unique_idx_for_all, :]

    # L2-normalize to use cosine similarity via dot product
    if used_encoder == 'tfidf':
        # Already normalized above
        pass
    else:
        norms = np.linalg.norm(embeddings_all, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        embeddings_all = embeddings_all / norms

    # Clustering
    algorithm = (algorithm or 'hdbscan').lower()
    labels = None

    if algorithm == 'hdbscan':
        try:
            import hdbscan
            # Use cosine if supported; else euclidean on normalized vectors
            metric = 'cosine' if similarity_metric == 'cosine' else 'euclidean'
            clusterer = hdbscan.HDBSCAN(min_cluster_size=max(2, int(min_cluster_size)), metric=metric)
            labels = clusterer.fit_predict(embeddings_all)
        except Exception:
            # Fallback to agglomerative average linkage with cosine
            from sklearn.cluster import AgglomerativeClustering
            kwargs = dict(linkage='average')
            # New sklearn uses 'metric', older uses 'affinity'
            try:
                if distance_threshold is not None:
                    clusterer = AgglomerativeClustering(n_clusters=None, distance_threshold=float(distance_threshold), metric='cosine', **kwargs)
                else:
                    if not num_clusters:
                        # Heuristic: sqrt(N)
                        num_clusters = max(2, int(np.sqrt(len(phrases))))
                    clusterer = AgglomerativeClustering(n_clusters=int(num_clusters), metric='cosine', **kwargs)
            except TypeError:
                if distance_threshold is not None:
                    clusterer = AgglomerativeClustering(n_clusters=None, distance_threshold=float(distance_threshold), affinity='cosine', **kwargs)
                else:
                    if not num_clusters:
                        num_clusters = max(2, int(np.sqrt(len(phrases))))
                    clusterer = AgglomerativeClustering(n_clusters=int(num_clusters), affinity='cosine', **kwargs)
            labels = clusterer.fit_predict(embeddings_all)

    elif algorithm == 'agglomerative':
        from sklearn.cluster import AgglomerativeClustering
        kwargs = dict(linkage='average')
        try:
            if distance_threshold is not None:
                clusterer = AgglomerativeClustering(n_clusters=None, distance_threshold=float(distance_threshold), metric='cosine', **kwargs)
            else:
                if not num_clusters:
                    num_clusters = max(2, int(np.sqrt(len(phrases))))
                clusterer = AgglomerativeClustering(n_clusters=int(num_clusters), metric='cosine', **kwargs)
        except TypeError:
            if distance_threshold is not None:
                clusterer = AgglomerativeClustering(n_clusters=None, distance_threshold=float(distance_threshold), affinity='cosine', **kwargs)
            else:
                if not num_clusters:
                    num_clusters = max(2, int(np.sqrt(len(phrases))))
                clusterer = AgglomerativeClustering(n_clusters=int(num_clusters), affinity='cosine', **kwargs)
        labels = clusterer.fit_predict(embeddings_all)

    elif algorithm == 'kmeans':
        from sklearn.cluster import KMeans
        if not num_clusters:
            num_clusters = max(2, int(np.sqrt(len(phrases))))
        try:
            clusterer = KMeans(n_clusters=int(num_clusters), n_init='auto', random_state=random_state)
        except TypeError:
            clusterer = KMeans(n_clusters=int(num_clusters), n_init=10, random_state=random_state)
        labels = clusterer.fit_predict(embeddings_all)

    elif algorithm == 'dbscan':
        from sklearn.cluster import DBSCAN
        # Heuristic eps for cosine distance on normalized vectors
        eps = 0.15
        clusterer = DBSCAN(eps=eps, min_samples=max(2, int(min_cluster_size)), metric='cosine')
        labels = clusterer.fit_predict(embeddings_all)
    else:
        raise ValueError("algorithm must be one of: 'hdbscan', 'agglomerative', 'kmeans', 'dbscan'")

    labels = np.asarray(labels, dtype=int)

    # Compute centroids per cluster (exclude noise label -1)
    unique_labels = sorted([l for l in np.unique(labels) if l != -1])
    centroids = {}
    for lab in unique_labels:
        idx = np.where(labels == lab)[0]
        if used_encoder == 'tfidf':
            # Mean of sparse rows
            centroid = embeddings_all[idx].mean(axis=0)
            # Normalize
            from sklearn.preprocessing import normalize as sk_normalize
            centroid = sk_normalize(centroid, norm='l2')
            centroids[lab] = np.asarray(centroid.toarray()).ravel()
        else:
            centroid = embeddings_all[idx].mean(axis=0)
            norm = np.linalg.norm(centroid)
            centroids[lab] = (centroid / norm) if norm > 0 else centroid

    # Compute distances to centroid (cosine distance = 1 - dot if normalized)
    distance_to_centroid = np.full(len(phrases), np.nan, dtype=float)
    for lab in unique_labels:
        idx = np.where(labels == lab)[0]
        c = centroids[lab]
        if used_encoder == 'tfidf':
            # embeddings_all[idx] is sparse; compute dot then 1 - cos
            dots = embeddings_all[idx].dot(c.reshape(-1, 1)).A.ravel()
        else:
            dots = embeddings_all[idx].dot(c)
        distance_to_centroid[idx] = 1.0 - np.clip(dots, -1.0, 1.0)

    # Pick representatives (min distance; tiebreak by shorter phrase then earlier order)
    representatives = set()
    for lab in unique_labels:
        idx = np.where(labels == lab)[0]
        if idx.size == 0:
            continue
        dists = distance_to_centroid[idx]
        # Tie-breakers
        lengths = np.array([len(phrases[i]) for i in idx])
        order = np.lexsort((np.arange(idx.size), lengths, dists))
        rep_idx = idx[order[0]]
        representatives.add(int(rep_idx))

    # Optional 2D projection
    x = np.full(len(phrases), np.nan, dtype=float)
    y = np.full(len(phrases), np.nan, dtype=float)
    if reduce_to_2d:
        if reducer == 'umap':
            try:
                import umap
                reducer_model = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1, metric='cosine', random_state=random_state)
                emb_2d = reducer_model.fit_transform(embeddings_all if used_encoder != 'tfidf' else embeddings_all.toarray())
                x, y = emb_2d[:, 0], emb_2d[:, 1]
            except Exception:
                reducer = 'pca'  # fallback
        if reducer == 'pca':
            from sklearn.decomposition import PCA
            pca = PCA(n_components=2, random_state=random_state)
            emb_matrix = embeddings_all if used_encoder != 'tfidf' else embeddings_all.toarray()
            try:
                emb_2d = pca.fit_transform(emb_matrix)
                x, y = emb_2d[:, 0], emb_2d[:, 1]
            except Exception:
                pass

    # Build output DataFrames
    clusters_df = pd.DataFrame({
        'phrase': phrases,
        'cluster_id': labels.astype(int),
        'distance_to_centroid': distance_to_centroid,
        'is_representative': [i in representatives for i in range(len(phrases))],
        'x': x,
        'y': y,
    })

    # Summarize clusters
    info_rows = []
    for lab in unique_labels:
        idx = np.where(labels == lab)[0]
        if idx.size == 0:
            continue
        # Representative phrase
        rep_candidates = clusters_df.iloc[idx].sort_values(['is_representative', 'distance_to_centroid'], ascending=[False, True])
        rep_phrase = rep_candidates.iloc[0]['phrase'] if not rep_candidates.empty else phrases[int(idx[0])]
        # Example phrases (up to 5)
        examples = clusters_df.iloc[idx].sort_values('distance_to_centroid').head(5)['phrase'].tolist()
        info_rows.append({
            'cluster_id': int(lab),
            'size': int(idx.size),
            'representative_phrase': rep_phrase,
            'centroid': centroids.get(lab, None),
            'example_phrases': examples,
        })
    cluster_info = pd.DataFrame(info_rows).sort_values('size', ascending=False).reset_index(drop=True)

    # Optionally attach embeddings
    if return_embeddings:
        try:
            if used_encoder == 'tfidf':
                emb_list = [embeddings_all[i].toarray().ravel().tolist() for i in range(embeddings_all.shape[0])]
            else:
                emb_list = embeddings_all.tolist()
            clusters_df['embedding'] = emb_list
        except Exception:
            pass

    return clusters_df, cluster_info


# ------------------------------------------------------------
# Utility: add frequency ranks to a DataFrame
# ------------------------------------------------------------
def add_frequency_rank(
    df,
    *,
    freq_col: str = 'frequency',
    group_by=None,                 # None | str | List[str]
    rank_col: str = 'frequency_rank',
    method: str = 'dense',         # 'average', 'min', 'max', 'first', 'dense', 'sequential'
    ascending: bool = False,       # False → highest frequency gets rank 1
    inplace: bool = False
):
    """
    Compute ranks based on a frequency-like column and append as a new column.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    freq_col : str
        Column to rank by (default 'frequency').
    group_by : None | str | List[str]
        If provided, ranks are computed within each group.
    rank_col : str
        Name of the new column to write (default 'frequency_rank').
    method : str
        Ranking method: 'average', 'min', 'max', 'first', 'dense', or 'sequential'.
        - 'sequential' assigns strict consecutive integers 1..N in sorted order
          (globally or within each group), ignoring ties.
    ascending : bool
        If True, lowest frequency → rank 1. Default False (highest → 1).
    inplace : bool
        If True, modify df in place and return it; otherwise return a copy.

    Returns
    -------
    pd.DataFrame
        DataFrame with the added rank column.
    """
    import pandas as pd

    if df is None or len(df) == 0:
        raise ValueError('DataFrame is empty')
    if freq_col not in df.columns:
        raise ValueError(f"Column {freq_col!r} not found in DataFrame")

    target = df if inplace else df.copy()

    # Validate group_by columns if provided
    if method == 'sequential':
        # Strict 1..N ranks by sorted order (global or within groups)
        if group_by is None:
            ordered_idx = target.sort_values(freq_col, ascending=ascending).index
            seq = pd.Series(range(1, len(ordered_idx) + 1), index=ordered_idx, dtype='int64')
            target[rank_col] = seq.loc[target.index].values
        else:
            gb_cols = [group_by] if isinstance(group_by, str) else list(group_by)
            missing = [c for c in gb_cols if c not in target.columns]
            if missing:
                raise ValueError(f"group_by columns not found in DataFrame: {missing}")
            def _seq_rank(g):
                ordered_idx = g.sort_values(freq_col, ascending=ascending).index
                return pd.Series(range(1, len(ordered_idx) + 1), index=ordered_idx, dtype='int64')
            seq = target.groupby(gb_cols, group_keys=False).apply(_seq_rank)
            target[rank_col] = 0
            target.loc[seq.index, rank_col] = seq.values
    else:
        if group_by is None:
            ranks = target[freq_col].rank(method=method, ascending=ascending)
        else:
            gb_cols = [group_by] if isinstance(group_by, str) else list(group_by)
            missing = [c for c in gb_cols if c not in target.columns]
            if missing:
                raise ValueError(f"group_by columns not found in DataFrame: {missing}")
            ranks = target.groupby(gb_cols)[freq_col].rank(method=method, ascending=ascending)

        # Cast to int for integral methods; leave as float for 'average'
        if method in ('dense', 'min', 'max', 'first'):
            try:
                ranks = ranks.astype(int)
            except Exception:
                pass
        target[rank_col] = ranks
    return target


# ------------------------------------------------------------
# Utility: normalize frequency to a new column
# ------------------------------------------------------------
def normalise_values(
    df,
    *,
    freq_col: str = 'frequency',
    out_col: str = 'normalised_values',
    method: str = 'max',          # 'max' | 'sum' | 'minmax' | 'zscore'
    group_by=None,                # None | str | List[str]
    inplace: bool = False,
    ddof: int = 0,                # for zscore std
    minmax_range=(0.0, 1.0)       # for minmax scaling
):
    """
    Create a normalized version of a frequency-like column.

    Methods
    -------
    - 'max':    x / max(x)
    - 'sum':    x / sum(x)
    - 'minmax': a + (x - min) * (b - a) / (max - min), with (a,b)=minmax_range
    - 'zscore': (x - mean) / std (ddof configurable)

    If group_by is provided, normalization is performed within each group.
    Division-by-zero is handled by returning 0.0 where the denominator is 0.
    """
    import pandas as pd
    import numpy as np

    if df is None or len(df) == 0:
        raise ValueError('DataFrame is empty')
    if freq_col not in df.columns:
        raise ValueError(f"Column {freq_col!r} not found in DataFrame")
    method = method.lower()
    if method not in ('max', 'sum', 'minmax', 'zscore'):
        raise ValueError("method must be one of: 'max', 'sum', 'minmax', 'zscore'")

    target = df if inplace else df.copy()

    if group_by is not None:
        gb_cols = [group_by] if isinstance(group_by, str) else list(group_by)
        missing = [c for c in gb_cols if c not in target.columns]
        if missing:
            raise ValueError(f"group_by columns not found in DataFrame: {missing}")
        g = target.groupby(gb_cols)
    else:
        g = None

    x = target[freq_col].astype(float)

    if method == 'max':
        denom = g[freq_col].transform('max').astype(float) if g is not None else float(x.max())
        with np.errstate(divide='ignore', invalid='ignore'):
            y = np.where(denom > 0, x / denom, 0.0)
    elif method == 'sum':
        denom = g[freq_col].transform('sum').astype(float) if g is not None else float(x.sum())
        with np.errstate(divide='ignore', invalid='ignore'):
            y = np.where(denom > 0, x / denom, 0.0)
    elif method == 'minmax':
        a, b = minmax_range
        x_min = g[freq_col].transform('min').astype(float) if g is not None else float(x.min())
        x_max = g[freq_col].transform('max').astype(float) if g is not None else float(x.max())
        rng = x_max - x_min
        with np.errstate(divide='ignore', invalid='ignore'):
            y = np.where(rng > 0, a + (x - x_min) * (b - a) / rng, a)
    else:  # 'zscore'
        mean = g[freq_col].transform('mean').astype(float) if g is not None else float(x.mean())
        if g is not None:
            # groupwise std with ddof
            std = g[freq_col].transform(lambda s: s.astype(float).std(ddof=ddof))
        else:
            std = float(x.std(ddof=ddof))
        with np.errstate(divide='ignore', invalid='ignore'):
            y = np.where(std > 0, (x - mean) / std, 0.0)

    target[out_col] = y
    return target


# ------------------------------------------------------------
# Fully parametric n-gram scatter with flexible x/y/color encodings
# ------------------------------------------------------------
def create_parametric_ngram_scatter(
    df,
    *,
    x_col: str,
    y_col: str,
    color_col: str = None,
    # How to interpret each channel
    x_mode: str = 'auto',           # 'categorical' | 'continuous' | 'auto'
    y_mode: str = 'auto',           # 'categorical' | 'continuous' | 'auto'
    color_mode: str = 'auto',       # 'categorical' | 'continuous' | 'auto'
    # Binning controls (only used for continuous channels)
    x_bin: bool = False,
    y_bin: bool = False,
    color_bin: bool = False,
    # If binning is enabled, choose bin spacing: 'linear' or 'log'
    x_bin_mode: str = 'linear',     # 'linear' | 'log'
    y_bin_mode: str = 'linear',     # 'linear' | 'log'
    color_bin_mode: str = 'linear', # 'linear' | 'log'
    x_bins: int = 20,
    y_bins: int = 20,
    color_bins: int = 8,
    # Axis/scale controls for unbinned continuous channels
    x_scale: str = 'linear',        # 'linear' | 'log'
    y_scale: str = 'linear',        # 'linear' | 'log'
    color_scale_treatment: str = 'auto',  # 'categorical' | 'continuous' | 'auto' (palette treatment)
    # Color palette controls
    color_scheme: str = None,       # Altair/Vega scheme name, e.g., 'tableau10', 'category20', 'viridis'
    color_range: list = None,       # Custom list of colors
    # Marks and layout
    size: int = 40,
    opacity: float = 0.6,
    width: int = 800,
    height: int = 550,
    tooltip: list = None,
    show_zero_axes: bool = False,
    tick_fontsize: int = 11,
    label_fontsize: int = 12,
):
    """
    Create a fully parametric Altair scatter for n-grams where x, y, and color
    encodings are user-defined and each channel can be treated as categorical or
    continuous, optionally binned with linear or logarithmic spacing.

    Notes
    -----
    - If a channel is categorical, no binning or normalization is applied.
    - If a channel is continuous:
        * Set `<channel>_bin=True` to discretize values into `maxbins=<channel>_bins`.
        * `*_bin_mode='log'` bins on log10(value). For axes, labels reflect log10.
        * If not binned, `*_scale='log'` applies a logarithmic axis scale.
    - Color can be categorical, continuous, or continuous+binned. Palette can be
      controlled via `color_scheme` or `color_range`. `color_scale_treatment`
      governs whether the palette behaves categorically or continuously.
    - If `show_zero_axes=True`, draw light guide rules at x=0 and/or y=0 when the
      corresponding axis is continuous, unbinned, and uses a linear scale (not log).
    """
    import pandas as pd
    import altair as alt

    if df is None or len(df) == 0:
        raise ValueError('DataFrame is empty')
    if x_col not in df.columns or y_col not in df.columns:
        raise ValueError(f"Missing required columns: {x_col!r} or {y_col!r}")
    if color_col is not None and color_col not in df.columns:
        raise ValueError(f"Color column {color_col!r} not found in DataFrame")

    data = df.copy()

    def _infer_mode(series, requested):
        if requested in ('categorical', 'continuous'):
            return requested
        return 'continuous' if pd.api.types.is_numeric_dtype(series) else 'categorical'

    x_mode_eff = _infer_mode(data[x_col], x_mode)
    y_mode_eff = _infer_mode(data[y_col], y_mode)
    color_mode_eff = None
    if color_col is not None:
        color_mode_eff = _infer_mode(data[color_col], color_mode)

    # Build optional transforms for log-binning
    log_calc_fields = []  # list of tuples (source_col, new_field)

    def _maybe_build_binned_field(col, mode, do_bin, bin_mode, new_suffix):
        """Return (field_name_for_encoding, bin_kwargs_or_None, title_suffix)."""
        if mode != 'continuous' or not do_bin:
            return col, None, ''
        if bin_mode == 'log':
            new_field = f"{col}__log10_{new_suffix}"
            log_calc_fields.append((col, new_field))
            return new_field, alt.Bin(maxbins=color_bins if new_suffix == 'color' else (x_bins if new_suffix == 'x' else y_bins)), ' (log10 binned)'
        # linear binning
        return col, alt.Bin(maxbins=color_bins if new_suffix == 'color' else (x_bins if new_suffix == 'x' else y_bins)), ' (binned)'

    x_field, x_bin_param, x_title_suffix = _maybe_build_binned_field(x_col, x_mode_eff, x_bin, x_bin_mode, 'x')
    y_field, y_bin_param, y_title_suffix = _maybe_build_binned_field(y_col, y_mode_eff, y_bin, y_bin_mode, 'y')
    if color_col is not None:
        color_field, color_bin_param, color_title_suffix = _maybe_build_binned_field(color_col, color_mode_eff, color_bin, color_bin_mode, 'color')
    else:
        color_field, color_bin_param, color_title_suffix = None, None, ''

    # Build channel encodings
    def _scale_for(mode, scale_pref):
        if mode != 'continuous' or scale_pref == 'linear':
            return alt.Undefined
        return alt.Scale(type='log')

    if x_mode_eff == 'categorical':
        x_enc = alt.X(
            f'{x_field}:N',
            title=x_col.replace('_', ' ').title(),
            axis=alt.Axis(labelFontSize=tick_fontsize, titleFontSize=label_fontsize)
        )
    else:
        x_enc = alt.X(
            f'{x_field}:Q',
            title=(f"log10({x_col})" if (x_bin and x_bin_mode == 'log') else x_col.replace('_', ' ').title()) + x_title_suffix,
            bin=x_bin_param if x_bin_param is not None else alt.Undefined,
            scale=_scale_for('continuous', x_scale),
            axis=alt.Axis(labelFontSize=tick_fontsize, titleFontSize=label_fontsize)
        )

    if y_mode_eff == 'categorical':
        y_enc = alt.Y(
            f'{y_field}:N',
            title=y_col.replace('_', ' ').title(),
            axis=alt.Axis(labelFontSize=tick_fontsize, titleFontSize=label_fontsize)
        )
    else:
        y_enc = alt.Y(
            f'{y_field}:Q',
            title=(f"log10({y_col})" if (y_bin and y_bin_mode == 'log') else y_col.replace('_', ' ').title()) + y_title_suffix,
            bin=y_bin_param if y_bin_param is not None else alt.Undefined,
            scale=_scale_for('continuous', y_scale),
            axis=alt.Axis(labelFontSize=tick_fontsize, titleFontSize=label_fontsize)
        )

    color_enc = alt.Undefined
    mark_kwargs = {'size': size, 'opacity': opacity}

    if color_col is not None:
        scale_kwargs = {}
        # Support reversed palette names like 'viridis_r' or 'viridis-r'
        def _resolve_vega_scheme_name(name):
            if name is None:
                return None
            mapping = {
                'tab10': 'tableau10',
                'tab20': 'tableau20',
            }
            return mapping.get(name, name)

        def _parse_scheme_name(name):
            if not name:
                return None, False
            n = str(name).strip()
            reverse = False
            lower = n.lower()
            if lower.endswith('_r') or lower.endswith('-r'):
                reverse = True
                n = n[:-2]
            n = _resolve_vega_scheme_name(n)
            return n, reverse

        if color_scheme:
            _scheme_name, _reverse = _parse_scheme_name(color_scheme)
            if _scheme_name:
                scale_kwargs['scheme'] = _scheme_name
            if _reverse:
                scale_kwargs['reverse'] = True
        if color_range:
            scale_kwargs['range'] = color_range

        # Determine how palette is treated (categorical vs continuous)
        palette_treatment = color_scale_treatment
        if palette_treatment == 'auto':
            palette_treatment = 'continuous' if (color_mode_eff == 'continuous' and not color_bin) else 'categorical'

        legend_title = color_col.replace('_', ' ').title() + color_title_suffix

        if color_mode_eff == 'categorical':
            color_enc = alt.Color(
                f'{color_field}:N',
                legend=alt.Legend(title=legend_title, labelFontSize=tick_fontsize, titleFontSize=label_fontsize),
                scale=alt.Scale(**scale_kwargs) if scale_kwargs else alt.Undefined
            )
        else:
            # Continuous data; either binned (discrete legend) or continuous scale
            if color_bin and color_bin_param is not None:
                color_enc = alt.Color(
                    f'{color_field}:Q',
                    bin=color_bin_param,
                    legend=alt.Legend(title=legend_title, labelFontSize=tick_fontsize, titleFontSize=label_fontsize),
                    scale=alt.Scale(**scale_kwargs) if scale_kwargs else alt.Undefined
                )
            else:
                # Unbinned continuous color
                color_enc = alt.Color(
                    f'{color_field}:Q',
                    legend=alt.Legend(title=legend_title, labelFontSize=tick_fontsize, titleFontSize=label_fontsize),
                    scale=alt.Scale(**scale_kwargs) if scale_kwargs else alt.Undefined
                )
    else:
        # Use a neutral fixed color if no color mapping is provided
        mark_kwargs['color'] = '#4c78a8'

    # Default tooltip
    if tooltip is None:
        tooltip = [
            alt.Tooltip('ngram:N', title='N-gram') if 'ngram' in data.columns else None,
            alt.Tooltip(f'{x_col}:{"Q" if x_mode_eff=="continuous" else "N"}', title=x_col.replace('_', ' ').title()),
            alt.Tooltip(f'{y_col}:{"Q" if y_mode_eff=="continuous" else "N"}', title=y_col.replace('_', ' ').title()),
        ]
        if color_col is not None:
            tooltip.append(alt.Tooltip(f'{color_col}:{"Q" if color_mode_eff=="continuous" else "N"}', title=color_col.replace('_', ' ').title()))
        # Remove Nones
        tooltip = [t for t in tooltip if t is not None]

    chart = alt.Chart(data)
    # Apply filters and calculations for log-binned channels
    for src_col, dst_col in log_calc_fields:
        chart = chart.transform_filter(f"datum['{src_col}'] > 0")
        chart = chart.transform_calculate(**{dst_col: f"log(datum['{src_col}'])/log(10)"})

    scatter = chart.mark_circle(**mark_kwargs).encode(
        x=x_enc,
        y=y_enc,
        color=color_enc,
        tooltip=tooltip
    ).properties(width=width, height=height)
    
    # Optionally add zero-axis guide lines when requested or when data span crosses zero
    layers = [scatter]
    try:
        # Only for continuous, unbinned axes and linear scales
        if x_mode_eff == 'continuous' and not x_bin and x_scale == 'linear':
            add_x_zero = False
            if show_zero_axes:
                add_x_zero = True
            else:
                x_min = pd.to_numeric(data[x_col], errors='coerce').min()
                x_max = pd.to_numeric(data[x_col], errors='coerce').max()
                if pd.notnull(x_min) and pd.notnull(x_max) and (x_min < 0) and (x_max > 0):
                    add_x_zero = True
            if add_x_zero:
                vline_df = pd.DataFrame({'zero': [0.0]})
                vline = alt.Chart(vline_df).mark_rule(color='#999999', opacity=0.6).encode(
                    x=alt.X('zero:Q')
                )
                layers.append(vline)
        if y_mode_eff == 'continuous' and not y_bin and y_scale == 'linear':
            add_y_zero = False
            if show_zero_axes:
                add_y_zero = True
            else:
                y_min = pd.to_numeric(data[y_col], errors='coerce').min()
                y_max = pd.to_numeric(data[y_col], errors='coerce').max()
                if pd.notnull(y_min) and pd.notnull(y_max) and (y_min < 0) and (y_max > 0):
                    add_y_zero = True
            if add_y_zero:
                hline_df = pd.DataFrame({'zero': [0.0]})
                hline = alt.Chart(hline_df).mark_rule(color='#999999', opacity=0.6).encode(
                    y=alt.Y('zero:Q')
                )
                layers.append(hline)
    except Exception:
        pass

    layered = alt.layer(*layers).resolve_scale(x='shared', y='shared')
    return layered


# ------------------------------------------------------------
# Fully parametric examples grid with n-gram texts in cells
# ------------------------------------------------------------
def ngram_examples_grid(
    df,
    *,
    x_col: str,
    y_col: str,
    x_mode: str = 'auto',            # 'categorical' | 'continuous' | 'auto'
    y_mode: str = 'auto',            # 'categorical' | 'continuous' | 'auto'
    # Binning for continuous axes
    x_binning: str = None,           # None | 'linear' | 'log' | 'signed_log'
    y_binning: str = None,           # None | 'linear' | 'log' | 'signed_log'
    x_bins: int = 8,
    y_bins: int = 8,
    # Category order/mapper for categorical axes
    x_categories=None,
    y_categories=None,
    x_category_mapper=None,
    y_category_mapper=None,
    # Examples selection
    examples_per_cell: int = 12,
    example_order_col: str = None,
    example_order_ascending: bool = False,
    wrap_width: int = 14,
    # Color for text
    color_col: str = None,
    color_mode: str = 'auto',        # 'categorical' | 'continuous' | 'auto'
    color_binning: str = None,       # None | 'linear' | 'log' (for continuous color)
    color_bins: int = 7,
    color_scheme: str = None,        # e.g., 'tab10', 'tab20', 'viridis' (matplotlib names). If 'tableau10'/'category20', auto-map to 'tab10'/'tab20'.
    color_range: list = None,        # explicit list of colors
    color_categories=None,           # explicit order for categorical color values
    # Styling
    figsize=None,
    facecolor: str = '#f8f9fb',
    cell_fill: str = '#f1f3f6',
    cell_fill_alpha: float = 0.35,
    cell_edge: str = '#d0d4da',
    tick_fontsize: int = 11,
    label_fontsize: int = 12,
    cell_text_fontsize: int = 11,
    x_label: str = None,
    y_label: str = None,
    show_zero_axes: bool = False,
):
    """
    Create a grid where columns are x (categories or bins) and rows are y (categories or bins),
    and each cell contains example n-gram texts taken from the subset of rows that fall into that cell.

    Returns
    -------
    table_df : pd.DataFrame
        One row per grid cell with example texts and counts. Includes bin ranges if applicable.
    fig : matplotlib.figure.Figure
        The rendered grid figure.
    """
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    import textwrap as _tw
    from matplotlib import cm
    from matplotlib.colors import ListedColormap, LinearSegmentedColormap, Normalize, LogNorm, BoundaryNorm

    if df is None or len(df) == 0:
        raise ValueError('DataFrame is empty')
    for col in [x_col, y_col]:
        if col not in df.columns:
            raise ValueError(f"Column {col!r} not found in DataFrame")
    if 'ngram' not in df.columns:
        raise ValueError("DataFrame must include 'ngram' column")
    if color_col is not None and color_col not in df.columns:
        raise ValueError(f"Color column {color_col!r} not found in DataFrame")

    data = df.copy()

    def _infer_mode(series, requested):
        if requested in ('categorical', 'continuous'):
            return requested
        return 'continuous' if pd.api.types.is_numeric_dtype(series) else 'categorical'

    x_mode_eff = _infer_mode(data[x_col], x_mode)
    y_mode_eff = _infer_mode(data[y_col], y_mode)

    # Prepare X axis structure (bins or categories)
    x_edges = None
    if x_mode_eff == 'categorical':
        x_vals = data[x_col] if x_category_mapper is None else data[x_col].apply(x_category_mapper)
        data['x_cat'] = x_vals
        if x_categories is None:
            uniq = pd.Series(x_vals.unique()).tolist()
            x_order = sorted(uniq, key=lambda v: (isinstance(v, str), v))
        else:
            x_order = x_categories
        x_labels = [str(v) for v in x_order]
        n_x = len(x_order)
    else:
        if x_binning not in ('linear', 'log', 'signed_log'):
            raise ValueError("For x continuous, x_binning must be 'linear', 'log', or 'signed_log'")
        x_series = data[x_col].astype(float)
        if x_binning == 'log':
            x_series = x_series[x_series > 0]
        if len(x_series) == 0:
            raise ValueError('No valid values for x after filtering for binning')
        if x_binning == 'log':
            xmin = max(1e-12, float(x_series.min()))
            xmax = float(x_series.max())
            if xmax < xmin:
                xmax = xmin
            x_edges = np.logspace(np.log10(xmin), np.log10(xmax), x_bins + 1)
        elif x_binning == 'signed_log':
            # Signed log binning: uniform in t-space where t = sign(x)*log10(1+|x|)
            t = np.sign(x_series) * np.log10(1.0 + np.abs(x_series))
            tmin = float(np.nanmin(t))
            tmax = float(np.nanmax(t))
            if not np.isfinite(tmin) or not np.isfinite(tmax):
                raise ValueError('No valid values for x after signed-log transform')
            if tmax == tmin:
                tmax = tmin + 1e-9
            t_edges = np.linspace(tmin, tmax, x_bins + 1)
            x_edges = np.sign(t_edges) * (np.power(10.0, np.abs(t_edges)) - 1.0)
        else:
            xmin = float(data[x_col].min())
            xmax = float(data[x_col].max())
            if xmax == xmin:
                xmax = xmin + 1e-9
            x_edges = np.linspace(xmin, xmax, x_bins + 1)
        # Build x labels
        def _fmt_edge(v):
            av = abs(v)
            if av >= 1:
                return f"{int(np.round(v))}"
            return f"{v:.2f}"
        x_labels = []
        for b in range(x_bins):
            left, right = x_edges[b], x_edges[b + 1]
            if x_binning == 'log':
                x_labels.append(f"{int(np.floor(left))}-{int(np.floor(right))}")
            elif x_binning == 'signed_log':
                x_labels.append(f"{_fmt_edge(left)}-{_fmt_edge(right)}")
            else:
                x_labels.append(f"{left:.2f}-{right:.2f}")
        x_order = list(range(x_bins))
        n_x = x_bins

    # Prepare Y axis structure (bins or categories)
    y_edges = None
    if y_mode_eff == 'categorical':
        y_vals = data[y_col] if y_category_mapper is None else data[y_col].apply(y_category_mapper)
        data['y_cat'] = y_vals
        if y_categories is None:
            uniq = pd.Series(y_vals.unique()).tolist()
            y_order = sorted(uniq, key=lambda v: (isinstance(v, str), v))
        else:
            y_order = y_categories
        y_labels = [str(v) for v in y_order]
        n_y = len(y_order)
    else:
        if y_binning not in ('linear', 'log', 'signed_log'):
            raise ValueError("For y continuous, y_binning must be 'linear', 'log', or 'signed_log'")
        y_series = data[y_col].astype(float)
        if y_binning == 'log':
            y_series = y_series[y_series > 0]
        if len(y_series) == 0:
            raise ValueError('No valid values for y after filtering for binning')
        if y_binning == 'log':
            ymin = max(1e-12, float(y_series.min()))
            ymax = float(y_series.max())
            if ymax < ymin:
                ymax = ymin
            y_edges = np.logspace(np.log10(ymin), np.log10(ymax), y_bins + 1)
        elif y_binning == 'signed_log':
            t = np.sign(y_series) * np.log10(1.0 + np.abs(y_series))
            tmin = float(np.nanmin(t))
            tmax = float(np.nanmax(t))
            if not np.isfinite(tmin) or not np.isfinite(tmax):
                raise ValueError('No valid values for y after signed-log transform')
            if tmax == tmin:
                tmax = tmin + 1e-9
            t_edges = np.linspace(tmin, tmax, y_bins + 1)
            y_edges = np.sign(t_edges) * (np.power(10.0, np.abs(t_edges)) - 1.0)
        else:
            ymin = float(data[y_col].min())
            ymax = float(data[y_col].max())
            if ymax == ymin:
                ymax = ymin + 1e-9
            y_edges = np.linspace(ymin, ymax, y_bins + 1)
        # Build y labels
        def _fmt_edge_y(v):
            av = abs(v)
            if av >= 1:
                return f"{int(np.round(v))}"
            return f"{v:.2f}"
        y_labels = []
        for b in range(y_bins):
            left, right = y_edges[b], y_edges[b + 1]
            if y_binning == 'log':
                y_labels.append(f"{int(np.floor(left))}-{int(np.floor(right))}")
            elif y_binning == 'signed_log':
                y_labels.append(f"{_fmt_edge_y(left)}-{_fmt_edge_y(right)}")
            else:
                y_labels.append(f"{left:.2f}-{right:.2f}")
        y_order = list(range(y_bins))
        n_y = y_bins

    # Determine example selection order
    if example_order_col is not None and example_order_col in data.columns:
        order_col = example_order_col
    else:
        order_col = None
        if x_mode_eff == 'continuous':
            order_col = x_col
        elif y_mode_eff == 'continuous':
            order_col = y_col
        elif 'frequency' in data.columns:
            order_col = 'frequency'

    # Prepare color mapping
    def _resolve_cmap_name(name):
        if name is None:
            return None
        mapping = {'tableau10': 'tab10', 'category20': 'tab20'}
        return mapping.get(name, name)

    cmap = None
    categorical_colors = None
    color_norm = None
    color_boundary_norm = None
    color_edges = None
    color_values_order = None

    if color_col is not None:
        # decide color mode
        if color_mode in ('categorical', 'continuous'):
            color_mode_eff = color_mode
        else:
            color_mode_eff = 'continuous' if pd.api.types.is_numeric_dtype(data[color_col]) else 'categorical'

        if color_mode_eff == 'categorical':
            if color_categories is not None:
                values = list(color_categories)
            else:
                raw_vals = pd.Series(data[color_col].unique()).tolist()
                # Default to sorted numeric if possible, else stable string sort
                try:
                    values = sorted(raw_vals)
                except Exception:
                    values = sorted(raw_vals, key=lambda v: (isinstance(v, str), v))
            color_values_order = list(values)
            # Build palette
            if color_range:
                palette = list(color_range)
            else:
                name = _resolve_cmap_name(color_scheme) or 'tab10'
                try:
                    base_cmap = cm.get_cmap(name)
                    # Prefer discrete indexing for ListedColormap (tab10/tableau10, etc.)
                    if hasattr(base_cmap, 'colors') and isinstance(getattr(base_cmap, 'colors'), (list, tuple)) and len(base_cmap.colors) > 0:
                        colors_list = list(base_cmap.colors)
                        palette = [colors_list[i % len(colors_list)] for i in range(len(values))]
                    else:
                        # Fallback to sampling across the range
                        palette = [base_cmap(i / max(1, len(values) - 1)) for i in range(len(values))]
                except Exception:
                    base_cmap = cm.get_cmap('tab10')
                    if hasattr(base_cmap, 'colors') and isinstance(getattr(base_cmap, 'colors'), (list, tuple)) and len(base_cmap.colors) > 0:
                        colors_list = list(base_cmap.colors)
                        palette = [colors_list[i % len(colors_list)] for i in range(len(values))]
                    else:
                        palette = [base_cmap(i / max(1, len(values) - 1)) for i in range(len(values))]
            categorical_colors = {val: palette[i % len(palette)] for i, val in enumerate(values)}
        else:
            # Continuous color
            series = data[color_col]
            # Colormap
            if color_range:
                cmap = LinearSegmentedColormap.from_list('custom', color_range)
            else:
                name = _resolve_cmap_name(color_scheme) or 'viridis'
                try:
                    cmap = cm.get_cmap(name)
                except Exception:
                    cmap = cm.get_cmap('viridis')

            if color_binning in ('linear', 'log'):
                if color_binning == 'log':
                    series = series[series > 0]
                if len(series) == 0:
                    # Fallback: treat as categorical with single color
                    color_boundary_norm = None
                else:
                    vmin = float(series.min())
                    vmax = float(series.max())
                    if color_binning == 'log':
                        color_edges = np.logspace(np.log10(max(1e-12, vmin)), np.log10(vmax), color_bins + 1)
                    else:
                        color_edges = np.linspace(vmin, vmax, color_bins + 1)
                    color_boundary_norm = BoundaryNorm(color_edges, ncolors=cmap.N)
            else:
                # Unbinned continuous
                series = data[color_col]
                vmin = float(series.min()) if len(series) else 0.0
                vmax = float(series.max()) if len(series) else 1.0
                if vmax == vmin:
                    vmax = vmin + 1e-9
                color_norm = Normalize(vmin=vmin, vmax=vmax)

    # Build table rows
    rows = []

    # Helper to filter by bin or category
    def _mask_for_x(sub):
        if x_mode_eff == 'categorical':
            return sub
        # continuous with bins: return same frame; we filter per-bin in the loop
        return sub

    # For figure size
    if figsize is None:
        w = max(10, n_x * 1.4)
        h = max(3.0, n_y * (1.2 + 0.35 * examples_per_cell))
        figsize = (w, h)

    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    ax.set_facecolor(facecolor)

    # Iterate rows and columns to build table and draw cells
    for r, y_key in enumerate(y_order):
        if y_mode_eff == 'categorical':
            row_df = data.loc[(data['y_cat'] == y_key)]
            y_label_display = str(y_key)
            y_left = y_right = None
            y_idx = r
        else:
            y_idx = y_key
            y_left, y_right = y_edges[y_idx], y_edges[y_idx + 1]
            # Filter per-bin (left inclusive only for first bin)
            if y_idx == 0:
                row_df = data.loc[(data[y_col] >= y_left) & (data[y_col] <= y_right)]
            else:
                row_df = data.loc[(data[y_col] > y_left) & (data[y_col] <= y_right)]
            y_label_display = y_labels[y_idx]

        for b, x_key in enumerate(x_order):
            if x_mode_eff == 'categorical':
                cell_df = row_df.loc[row_df['x_cat'] == x_key]
                x_label_display = str(x_key)
                x_left = x_right = None
                x_idx = b
            else:
                x_idx = x_key
                x_left, x_right = x_edges[x_idx], x_edges[x_idx + 1]
                if x_idx == 0:
                    cell_df = row_df.loc[(row_df[x_col] >= x_left) & (row_df[x_col] <= x_right)]
                else:
                    cell_df = row_df.loc[(row_df[x_col] > x_left) & (row_df[x_col] <= x_right)]
                x_label_display = x_labels[x_idx]

            # Sort for example selection
            if order_col is not None and order_col in cell_df.columns:
                cell_df = cell_df.sort_values(order_col, ascending=example_order_ascending)
            # If order_col missing or all NaN, fallback to original order

            # Pick examples and optional color values
            ex_texts = cell_df['ngram'].head(examples_per_cell).astype(str).tolist()
            if color_col is not None:
                ex_color_vals = cell_df[color_col].head(examples_per_cell).tolist()
            else:
                ex_color_vals = [None] * len(ex_texts)

            # Store table row
            rows.append({
                'y_key': y_key,
                'x_key': x_key,
                'y_idx': y_idx,
                'x_idx': x_idx,
                'y_label': y_label_display,
                'x_label': x_label_display,
                'y_left': y_left,
                'y_right': y_right,
                'x_left': x_left,
                'x_right': x_right,
                'count': int(cell_df.shape[0]),
                'examples': ex_texts,
                'examples_color_values': ex_color_vals,
            })

            # Draw cell rectangle
            cell_rect = Rectangle(
                (b, n_y - 1 - r), 1, 1,
                fill=True,
                facecolor=cell_fill,
                alpha=cell_fill_alpha,
                edgecolor=cell_edge,
                linewidth=1.0
            )
            ax.add_patch(cell_rect)

            # Lay out example texts
            pad = 0.08
            if examples_per_cell > 0:
                y_positions = np.linspace(pad, 1.0 - pad, num=examples_per_cell + 2, endpoint=True)[1:-1]
            else:
                y_positions = []
            for k, text in enumerate(ex_texts[:examples_per_cell]):
                xv = ex_color_vals[k] if k < len(ex_color_vals) else None
                # Resolve text color
                txt_color = 'black'
                if color_col is not None:
                    if categorical_colors is not None:
                        txt_color = categorical_colors.get(xv, 'black')
                    elif cmap is not None:
                        if color_boundary_norm is not None and (xv is not None):
                            try:
                                txt_color = cmap(color_boundary_norm(float(xv)))
                            except Exception:
                                txt_color = 'black'
                        elif color_norm is not None and (xv is not None):
                            try:
                                txt_color = cmap(color_norm(float(xv)))
                            except Exception:
                                txt_color = 'black'
                wrapped = _tw.fill(str(text), width=wrap_width)
                x_pos = b + pad
                y_pos = (n_y - 1 - r) + y_positions[k] if k < len(y_positions) else (n_y - 1 - r) + pad
                ax.text(x_pos, y_pos, wrapped, ha='left', va='bottom', fontsize=cell_text_fontsize, color=txt_color)

    # Optional zero-axis guide lines for continuous axes (linear or signed_log)
    if show_zero_axes:
        # Vertical line at x=0
        if x_mode_eff == 'continuous' and x_binning in ('linear', 'signed_log') and x_edges is not None:
            try:
                for b in range(n_x):
                    left, right = x_edges[b], x_edges[b + 1]
                    if (left <= 0.0 <= right) or (right <= 0.0 <= left):
                        frac = 0.0 if right == left else (0.0 - left) / (right - left)
                        x_zero = b + frac
                        ax.axvline(x_zero, color='#999999', alpha=0.6, linewidth=1.2)
                        break
            except Exception:
                pass
        # Horizontal line at y=0
        if y_mode_eff == 'continuous' and y_binning in ('linear', 'signed_log') and y_edges is not None:
            try:
                for b in range(n_y):
                    left, right = y_edges[b], y_edges[b + 1]
                    if (left <= 0.0 <= right) or (right <= 0.0 <= left):
                        frac = 0.0 if right == left else (0.0 - left) / (right - left)
                        y_zero = b + frac
                        ax.axhline(y_zero, color='#999999', alpha=0.6, linewidth=1.2)
                        break
            except Exception:
                pass

    # Ticks and labels
    ax.set_xticks(np.arange(n_x) + 0.5)
    ax.set_xticklabels(x_labels, rotation=90, fontsize=tick_fontsize)
    ax.set_yticks(np.arange(n_y) + 0.5)
    ax.set_yticklabels(list(reversed(y_labels)), fontsize=tick_fontsize)
    ax.set_xlim(0, n_x)
    ax.set_ylim(0, n_y)
    ax.invert_yaxis()
    ax.grid(False)

    # Axis labels
    ax.set_xlabel(x_label or (f"{x_col} (bin)" if x_mode_eff == 'continuous' else str(x_col).replace('_', ' ').title()), fontsize=label_fontsize)
    ax.set_ylabel(y_label or (f"{y_col} (bin)" if y_mode_eff == 'continuous' else str(y_col).replace('_', ' ').title()), fontsize=label_fontsize)

    table_df = pd.DataFrame(rows, columns=[
        'y_key','x_key','y_idx','x_idx','y_label','x_label','y_left','y_right','x_left','x_right','count','examples','examples_color_values'
    ])
    
    # Legend or colorbar when a color dimension is defined
    if color_col is not None:
        from matplotlib.lines import Line2D
        legend_title = str(color_col).replace('_', ' ').title()
        if categorical_colors is not None:
            # Categorical legend using discrete color patches
            handles = []
            # Preserve a stable order: user-provided color_categories > computed order > appearance order
            if color_categories is not None:
                legend_values = list(color_categories)
            elif color_values_order is not None:
                legend_values = list(color_values_order)
            else:
                seen = []
                for val in data[color_col].tolist():
                    if val not in seen:
                        seen.append(val)
                legend_values = seen
            for val in legend_values:
                colr = categorical_colors.get(val, 'black')
                handles.append(Line2D([0], [0], marker='s', color='none', markerfacecolor=colr, markersize=10, label=str(val)))
            legend = ax.legend(
                handles=handles,
                title=legend_title,
                loc='upper center',
                bbox_to_anchor=(0.5, -0.15),
                ncol=max(1, min(len(handles), 6)),
                frameon=False
            )
            if legend and legend.get_title() is not None:
                legend.get_title().set_fontsize(label_fontsize)
            if legend is not None:
                for text in legend.get_texts():
                    text.set_fontsize(tick_fontsize)
        elif cmap is not None:
            # Continuous colorbar (binned or unbinned)
            from matplotlib.cm import ScalarMappable
            if color_boundary_norm is not None:
                mappable = ScalarMappable(norm=color_boundary_norm, cmap=cmap)
                mappable.set_array([])
                cbar = fig.colorbar(mappable, ax=ax, orientation='horizontal', fraction=0.046, pad=0.20)
                cbar.set_label(legend_title, fontsize=label_fontsize)
                cbar.ax.tick_params(labelsize=tick_fontsize)
            elif color_norm is not None:
                mappable = ScalarMappable(norm=color_norm, cmap=cmap)
                mappable.set_array([])
                cbar = fig.colorbar(mappable, ax=ax, orientation='horizontal', fraction=0.046, pad=0.20)
                cbar.set_label(legend_title, fontsize=label_fontsize)
                cbar.ax.tick_params(labelsize=tick_fontsize)
            # Else: no valid norm; skip legend

        # Ensure extra bottom space so legend/colorbar does not overlap x-label
        try:
            fig.subplots_adjust(bottom=0.20)
        except Exception:
            pass

    return table_df, fig
