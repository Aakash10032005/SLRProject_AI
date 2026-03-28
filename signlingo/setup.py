from setuptools import setup, find_packages

setup(
    name='signlingo',
    version='1.0.0',
    description='Real-time ASL recognition with Indian language translation',
    author='SignLingo Team',
    license='Apache 2.0',
    packages=find_packages(),
    python_requires='>=3.11',
    install_requires=[
        'torch>=2.3.0',
        'torchvision>=0.18.0',
        'timm>=1.0.0',
        'mediapipe>=0.10.0',
        'opencv-python>=4.9.0',
        'PyQt6>=6.7.0',
        'requests>=2.31.0',
        'numpy>=1.26.0',
        'Pillow>=10.0.0',
        'pyyaml>=6.0.1',
        'python-dotenv>=1.0.0',
        'pyttsx3>=2.90',
        'scipy>=1.13.0',
        'tqdm>=4.66.0',
    ],
    entry_points={
        'console_scripts': ['signlingo=run:main'],
    },
)
