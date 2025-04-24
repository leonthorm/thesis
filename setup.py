from pathlib import Path
from setuptools import setup, find_packages

PROJECT_ROOT = Path(__file__).parent

imitation_path = PROJECT_ROOT / "deps" / "imitation"
dynobench_path = PROJECT_ROOT / "deps" / "dynobench"

setup(
    name="thesis",
    version="0.1",
    packages=find_packages(where='.'),
    install_requires=[
        f"imitation @ file://{imitation_path}",
        f"dynobench @ file://{dynobench_path}",
        'gymnasium==0.29.1',
        'mujoco==3.2.5',
        'stable-baselines3==2.2.1',
        'wandb==0.19.0',
        'scipy==1.10.1',
        'tensorboard==2.14.0',
        'numpy<=2.0.0',
        'numba==0.61.0',
        'tqdm==4.67.1',
        'rich==13.9.4',
        'rowan',
        'cvxpy',
        'imageio'
    ],
    entry_points={
        'console_scripts': [
            'train-dagger=scripts.train_dagger_coltrans_dyno:main',
            'validate-policy=scripts.validate_policy:main',
            'validate-dir=scripts.validate_dir:main',
            'visualize-payload=scripts.analysis.visualize_payload:main',
            # 'run-dagger-dbcbs=scripts.run_dagger_dbcbs:main',
        ],
    },
)
