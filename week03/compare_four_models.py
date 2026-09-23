"""Create a single test-set comparison figure for the four classroom models."""
import json
from collections import OrderedDict
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import trajnetplusplustools


ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / 'OUTPUT_BLOCK' / 'circle_classroom'
MODELS = OrderedDict([
    ('Social baseline', OUTPUT_DIR / 'lstm_social_baseline.pkl'),
    ('Same-cell mean', OUTPUT_DIR / 'lstm_social_social_mean.pkl'),
    ('Wider range', OUTPUT_DIR / 'lstm_social_wider_range.pkl'),
    ('Vanilla', OUTPUT_DIR / 'lstm_vanilla_no_neighbours.pkl'),
])


def load_first_test_scene():
    reader = trajnetplusplustools.Reader(
        str(ROOT / 'DATA_BLOCK' / 'circle_classroom' / 'test' / 'circle.ndjson'),
        scene_type='paths',
    )
    scene_id, paths = next(iter(reader.scenes()))
    xy = torch.tensor(trajnetplusplustools.Reader.paths_to_xy(paths), dtype=torch.float32)
    return scene_id, xy


def main():
    torch.set_num_threads(2)
    scene_id, xy = load_first_test_scene()
    history = xy[:8].clone()
    truth = xy[8:, 0].numpy()
    cv = (
        history[-1, 0]
        + torch.arange(1, 13)[:, None] * (history[-1, 0] - history[-2, 0])
    ).numpy()

    predictions = OrderedDict()
    metrics = OrderedDict()
    for label, checkpoint in MODELS.items():
        if not checkpoint.exists():
            raise FileNotFoundError(checkpoint)
        metrics_path = Path(str(checkpoint) + '.test.metrics.json')
        if not metrics_path.exists():
            raise FileNotFoundError(metrics_path)
        predictor = torch.load(checkpoint, map_location='cpu')
        model = predictor.model.eval()
        with torch.no_grad():
            _, positions = model(
                history,
                torch.zeros(xy.shape[1], 2),
                torch.tensor([0, xy.shape[1]]),
                n_predict=12,
            )
        predictions[label] = positions[-12:, 0].numpy()
        payload = json.loads(metrics_path.read_text(encoding='utf-8'))
        metrics[label] = (payload['model']['ADE'], payload['model']['FDE'])

    cv_payload = json.loads(
        Path(str(next(iter(MODELS.values()))) + '.test.metrics.json').read_text(encoding='utf-8')
    )
    cv_ade = cv_payload['constant_velocity']['ADE']
    cv_fde = cv_payload['constant_velocity']['FDE']

    colors = {
        'Social baseline': '#2ca02c',
        'Same-cell mean': '#9467bd',
        'Wider range': '#d62728',
        'Vanilla': '#8c564b',
    }
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), gridspec_kw={'width_ratios': [1.5, 1, 1]})

    ax = axes[0]
    ax.plot(history[:, 0, 0], history[:, 0, 1], '.-', linewidth=2.2, label='History', color='#1f77b4')
    ax.plot(*np.vstack([history[-1, 0].numpy(), truth]).T, '.-', linewidth=2.2,
            label='Truth', color='#ff7f0e')
    ax.plot(*np.vstack([history[-1, 0].numpy(), cv]).T, '.--', linewidth=1.7,
            label='Constant velocity', color='#7f7f7f')
    for label, prediction in predictions.items():
        path = np.vstack([history[-1, 0].numpy(), prediction])
        ax.plot(path[:, 0], path[:, 1], '.-', linewidth=1.8, label=label, color=colors[label])
    ax.set_title(f'Test scene {scene_id}: primary pedestrian')
    ax.set_xlabel('x (scaled units)')
    ax.set_ylabel('y (scaled units)')
    ax.axis('equal')
    ax.grid(alpha=0.2)
    ax.legend(fontsize=9)

    labels = list(metrics)
    x = np.arange(len(labels))
    short_labels = ['Baseline', 'Mean', 'Wider', 'Vanilla']
    for metric_index, (metric_name, cv_value) in enumerate((('ADE', cv_ade), ('FDE', cv_fde)), start=1):
        values = [metrics[label][metric_index - 1] for label in labels]
        bars = axes[metric_index].bar(x, values, color=[colors[label] for label in labels])
        axes[metric_index].axhline(cv_value, color='#7f7f7f', linestyle='--', linewidth=1.5,
                                  label=f'CV = {cv_value:.3f}')
        axes[metric_index].set_title(f'Test {metric_name} (lower is better)')
        axes[metric_index].set_xticks(x, short_labels, rotation=20)
        axes[metric_index].set_ylabel('Error (scaled units)')
        axes[metric_index].set_ylim(0, max(values) * 1.22)
        axes[metric_index].grid(axis='y', alpha=0.2)
        axes[metric_index].legend(fontsize=9)
        axes[metric_index].bar_label(bars, fmt='%.3f', padding=3, fontsize=9)

    fig.suptitle('Four-model comparison on the fixed classroom test split', fontsize=15)
    fig.tight_layout()
    asset_dir = ROOT / 'assets'
    asset_dir.mkdir(exist_ok=True)
    output = asset_dir / 'four_models_test_comparison.png'
    fig.savefig(output, dpi=180, bbox_inches='tight')
    plt.close(fig)
    print(output)


if __name__ == '__main__':
    main()
