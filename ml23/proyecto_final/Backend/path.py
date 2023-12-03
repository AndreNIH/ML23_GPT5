from pathlib import Path

file_path = Path(__file__).parent.absolute()

dataset_path = file_path / 'data' / 'Dataset'

if not dataset_path.exists():
    dataset_path.mkdir(parents=True, exist_ok=True)

print(f"Path to Dataset directory: {dataset_path}")