# MLOps Movie Recommendation

This project provides a complete MLOps pipeline for a movie recommendation system, leveraging modern tools such as Docker, Airflow, MLflow, Grafana, Prometheus, and Evidently.

## Prerequisites

Before running the project, please ensure the following steps are completed:

### 1. Create a `.env.local` file in the project root

This file must contain the following environment variables, adjusted to your environment:

```env
HOST_PROJECT_PATH=repo path in your local environment
GITHUB_USERNAME=your_github_username
GITHUB_EMAIL=your_github_email
GITHUB_TOKEN=your_github_token
AWS_ACCESS_KEY_ID=your_dagshub_access_key_id
AWS_SECRET_ACCESS_KEY=your_dagshub_secret_access_key
AIRFLOW_UID=1000
AIRFLOW_GID=0
DOCKER_GID=$(getent group docker | cut -d: -f3)
API_URL=http://api:8000
API_KEY=your_api_key
```

- **HOST_PROJECT_PATH**: The absolute path to your cloned repository (e.g., `/home/ubuntu/MLOps_movie_recommandation`).
- **DOCKER_GID**: You can obtain your Docker group id by running `getent group docker | cut -d: -f3` in your terminal.
- **API_KEY**: **You must request this from the project administrators.**

> **Important:**  
> - Replace all placeholder values (`your_*`) with your actual credentials.
> - Replace `your_dagshub_access_key_id` and `your_dagshub_secret_access_key` with your Dagshub credentials.
> - To obtain the API key, please **request it from the project administrators**.
> - You **must also ask the repository administrators for read/write access to this repository** in order for the project to work fully.

### 2. Run the Initialization Script

From the project root, execute:

```bash
bash src/init.sh
```

This script will:
- Check for the `.env.local` file
- Set up the required Docker network
- Build Docker images
- Start all necessary services:
  - Data import service
  - API (FastAPI)
  - MySQL
  - MLflow server
  - Evidently
  - Prometheus
  - Grafana
- Optionally launch Airflow (you will be prompted)

## Services and Interfaces

Once initialized, you can access the following interfaces:

| Service       | URL                            | Notes                        |
|---------------|------------------------------- |-----------------------------|
| API           | http://localhost:8000/health   | FastAPI health endpoint      |
| MLflow        | http://localhost:5050          | ML experiment tracking       |
| Grafana       | http://localhost:3000          | Dashboards and monitoring    |
| Prometheus    | http://localhost:9090          | Metrics scraping             |
| Airflow       | http://localhost:8080          | Orchestration (login: airflow / password: airflow) |

## Project Structure

```
.
├── src/
│   └── init.sh        # Initialization script (see above)
├── .env.local         # Environment variables (must create)
├── docker-compose.yml
├── docker-compose.airflow.yml
├── ...                # Other source files
```

## Troubleshooting

- If any service does not start, review the logs printed by `init.sh`.
- Ensure Docker is installed and running on your machine.
- Verify all required environment variables are set properly.
- For Airflow, if you choose not to start it during initialization, you can always start it manually using:
  ```bash
  docker-compose --env-file .env.local -f docker-compose.airflow.yml up -d
  ```

## License

This project is provided under the MIT License.

---

Feel free to contribute or open issues for questions and improvements!