import os
import subprocess
import argparse
import shutil
import sys
from pathlib import Path

# Allow COLMAP (Qt-based) to run on headless servers without an X display.
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

def find_colmap_executable():
    """Find COLMAP from PATH or from the active conda environment."""
    colmap = shutil.which('colmap')
    if colmap is not None:
        return colmap

    prefixes = [os.path.dirname(sys.executable)]
    conda_prefix = os.environ.get('CONDA_PREFIX')
    if conda_prefix:
        prefixes.append(conda_prefix)

    candidates = []
    for prefix in prefixes:
        candidates.extend([
            os.path.join(prefix, 'Library', 'bin', 'colmap.exe'),
            os.path.join(prefix, 'bin', 'colmap'),
        ])
    for candidate in candidates:
        if os.path.exists(candidate):
            bin_dir = os.path.dirname(candidate)
            env_root = os.path.dirname(os.path.dirname(bin_dir)) if os.path.basename(bin_dir).lower() == 'bin' else os.path.dirname(sys.executable)
            path_parts = [
                env_root,
                os.path.join(env_root, 'Library', 'mingw-w64', 'bin'),
                os.path.join(env_root, 'Library', 'usr', 'bin'),
                os.path.join(env_root, 'Library', 'bin'),
                os.path.join(env_root, 'Scripts'),
            ]
            os.environ['PATH'] = os.pathsep.join(path_parts + [os.environ.get('PATH', '')])
            return candidate

    raise FileNotFoundError(
        "COLMAP executable not found. Install it into the active environment "
        "or run via `conda run -n dip ...`."
    )

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Run COLMAP for multi-view stereo')
    parser.add_argument('--data_dir', type=str, required=True, help='Path to the input directory containing images in data_dir/images')
    parser.add_argument('--results_dir', type=str, default=None, help='Optional root-level copy directory for TXT model files (default: results/<scene>/colmap)')
    args = parser.parse_args()
    data_dir = args.data_dir
    colmap = find_colmap_executable()

    # Feature extraction with shared intrinsics (assume it's the same camera)
    subprocess.run([colmap, 'feature_extractor', '--image_path', os.path.join(data_dir, 'images'), '--database_path', os.path.join(data_dir, 'database.db'), '--ImageReader.single_camera', '1', '--ImageReader.camera_model', 'PINHOLE', '--SiftExtraction.use_gpu', '0'], check=True)

    # Feature matching
    subprocess.run([colmap, 'exhaustive_matcher', '--database_path', os.path.join(data_dir, 'database.db'), '--SiftMatching.use_gpu', '0'], check=True)

    # Create sparse reconstruction folder
    os.makedirs(os.path.join(data_dir, 'sparse'), exist_ok=True)

    # Sparse reconstruction
    subprocess.run([colmap, 'mapper', '--image_path', os.path.join(data_dir, 'images'), '--database_path', os.path.join(data_dir, 'database.db'), '--output_path', os.path.join(data_dir, 'sparse')], check=True)

    # Convert binary model to text format
    os.makedirs(os.path.join(data_dir, 'sparse', '0_text'), exist_ok=True)
    text_model_dir = os.path.join(data_dir, 'sparse', '0_text')
    subprocess.run([colmap, 'model_converter', '--input_path', os.path.join(data_dir, 'sparse', '0'), '--output_path', text_model_dir, '--output_type', 'TXT'], check=True)

    scene_name = os.path.basename(os.path.normpath(data_dir))
    results_dir = args.results_dir or os.path.join('results', scene_name, 'colmap')
    Path(results_dir).mkdir(parents=True, exist_ok=True)
    for name in ('cameras.txt', 'images.txt', 'points3D.txt'):
        src = os.path.join(text_model_dir, name)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(results_dir, name))

    print("COLMAP multi-view stereo pipeline completed successfully!")
    print("Sparse 3D reconstruction saved in:", text_model_dir)
    print("A copy for submission/reference was saved in:", results_dir)
    
