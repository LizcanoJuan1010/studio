# Fruit Model Comparator - Next.js & FastAPI

This project is a web application to visualize and compare the performance of pre-trained machine learning models for fruit image classification. It features a Next.js frontend and a Python FastAPI backend to serve the models.

## Architectural Overview

This application is set up as a multi-container application managed by Docker Compose.

-   **Frontend:** A Next.js application responsible for rendering the user interface. It's built with React, TypeScript, and styled using **Tailwind CSS** and **ShadCN UI** components.
-   **Backend:** A FastAPI service written in Python. This backend is responsible for loading the machine learning models (TensorFlow/Keras, Scikit-learn) and exposing a prediction endpoint. This separation ensures that the heavy model inference does not block the web server.
-   **Containerization:** Both the frontend and backend are containerized and managed by Docker. `docker compose` orchestrates the building and running of both services, including the network communication between them.
-   **Data Flow:** The Next.js frontend sends image data to the FastAPI backend for inference. The backend runs the predictions on all models and returns the results to the frontend for visualization.

## How to Run the Application

### Prerequisites

-   [Docker](https://www.docker.com/get-started) and Docker Compose (v2 recommended - use `docker compose` command)

### Running with Docker

This is the recommended way to run the project.

1.  **Build the Docker images:**
    Open your terminal in the project root and run:
    ```bash
    docker compose build
    ```

2.  **Start the application:**
    ```bash
    docker compose up
    ```
    You can add the `-d` flag (`docker compose up -d`) to run the containers in the background.

3.  **Access the application:**
    -   **Frontend:** The web application will be available at [http://localhost:3000](http://localhost:3000).
    -   **Backend API (for testing):** The FastAPI backend is available at [http://localhost:8001](http://localhost:8001). You can visit `http://localhost:8001/docs` for the API documentation.

## Project Structure

-   `docker-compose.yml`: Configuration file for orchestrating the frontend and backend services.
-   `Dockerfile`: Located in the root, this file defines the container for the **Next.js frontend**.
-   `backend/`: Contains all the code for the Python API.
    -   `main.py`: The FastAPI application that loads the models and defines the `/predict/` endpoint.
    -   `Dockerfile`: Defines the container for the **FastAPI backend**, including Python and library installations.
    -   `requirements.txt`: Lists the Python dependencies for the backend.
    -   `models/`: Contains the pre-trained model files (`.keras`, `.joblib`, `.pkl`).
-   `src/app/`: The Next.js frontend application.
    -   `(main)/`: Contains the pages for the dashboard (Home, Compare, Live Test, etc.).
    -   `actions.ts`: Contains the Server Action that communicates with the FastAPI backend to get predictions.
-   `src/components/`: Reusable React components for the UI.
-   `src/data/`: JSON data files for model metadata and evaluation metrics (e.g., accuracy, F1-score).
-   `src/lib/`: Core TypeScript logic and type definitions for the frontend.