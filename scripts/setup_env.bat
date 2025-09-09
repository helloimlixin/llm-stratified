@echo off
REM Setup script for Stratified Manifold Learning project (Windows)
REM This script creates a conda environment and installs all dependencies

echo 🚀 Setting up Stratified Manifold Learning environment...

REM Check if conda is installed
where conda >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo ❌ Conda is not installed. Please install Anaconda or Miniconda first.
    echo    Download from: https://docs.conda.io/en/latest/miniconda.html
    pause
    exit /b 1
)

REM Check if environment already exists
conda env list | findstr "stratified-manifold-learning" >nul
if %ERRORLEVEL% EQU 0 (
    echo ⚠️  Environment 'stratified-manifold-learning' already exists.
    set /p choice="Do you want to remove it and create a new one? (y/N): "
    if /i "%choice%"=="y" (
        echo 🗑️  Removing existing environment...
        conda env remove -n stratified-manifold-learning -y
    ) else (
        echo ℹ️  Using existing environment. Activating...
        call conda activate stratified-manifold-learning
        echo ✅ Environment activated!
        pause
        exit /b 0
    )
)

REM Create conda environment from environment.yml
echo 📦 Creating conda environment from environment.yml...
conda env create -f environment.yml

REM Activate the environment
echo 🔄 Activating environment...
call conda activate stratified-manifold-learning

REM Verify installation
echo 🔍 Verifying installation...
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
python -c "import transformers; print(f'Transformers version: {transformers.__version__}')"
python -c "import sklearn; print(f'Scikit-learn version: {sklearn.__version__}')"

REM Check CUDA availability
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import torch; torch.cuda.is_available()" >nul 2>nul
if %ERRORLEVEL% EQU 0 (
    python -c "import torch; print(f'CUDA version: {torch.version.cuda}')"
)

echo.
echo ✅ Setup complete!
echo.
echo 📋 Next steps:
echo    1. Activate the environment: conda activate stratified-manifold-learning
echo    2. Run a quick test: python example.py
echo    3. Run experiments: python main.py --model roberta --samples-per-domain 100
echo.
echo 🔧 Environment info:
python --version
echo    Environment name: stratified-manifold-learning
echo.
echo 💡 Tips:
echo    - Always activate the environment before working: conda activate stratified-manifold-learning
echo    - To deactivate: conda deactivate
echo    - To remove environment: conda env remove -n stratified-manifold-learning

pause
