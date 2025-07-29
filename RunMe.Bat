@echo off
SETLOCAL

REM --- Aller dans le dossier contenant ce script .bat ---
cd /d "%~dp0"

REM --- Création de l'environnement virtuel si nécessaire ---
if not exist "venv" (
    echo Création de l'environnement virtuel...
    python -m venv venv
)

REM --- Activation de l'environnement virtuel ---
call venv\Scripts\activate

REM --- Installation des dépendances ---
echo Installation des dépendances Python...
pip install --upgrade pip
pip install -r requirements.txt

REM --- Lancement du script ---
echo Lancement de l'application...
python app.py

pause

