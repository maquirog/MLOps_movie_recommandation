import os
from fastapi import HTTPException
import time
import requests
import sys

def validate_file_path(file_path: str, file_description: str):
    """
    Valide si un fichier existe et soulève une exception HTTP si ce n'est pas le cas.

    Args:
        file_path (str): Chemin vers le fichier.
        file_description (str): Description du fichier pour le message d'erreur.

    Raises:
        HTTPException: Si le fichier n'existe pas.
    """
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail=f"{file_description} non trouvé: {file_path}")

# Fonction que l'on pourrait utiliser pour le dag automatic_retrain, en tant que 2eme task après le lancement de l'API.
# Pour s'assurer que l'API est prête avant de lancer les tâches suivantes.
# Nécessite de créer une image Docker car il faut être sur le network `mlops-net` pour communiquer avec l'API.
def wait_for_api():
    url = "http://0.0.0.0:8000"
    print("Waiting for API to be ready...")
    for _ in range(30):  # Réessayer pendant 30 secondes
        try:
            response = requests.get(url)
            if response.status_code == 200:
                print("API is ready!")
                return
        except requests.ConnectionError:
            pass
        time.sleep(1)
    print("API not ready after 30 seconds.")
    sys.exit(1)

if __name__ == "__main__":
    wait_for_api()