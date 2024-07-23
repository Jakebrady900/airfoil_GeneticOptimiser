import GeneticAlgorithm as GA
import threading
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles  # Import StaticFiles

App = FastAPI()

# Configure CORS middleware
App.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# Mount a directory to serve static files (images)
App.mount("/outputs", StaticFiles(directory="outputs"), name="outputs") 


@App.post("/run")
async def HandleRequest(payload: dict):
    if "solution_type" not in payload or "velocity" not in payload:
        raise HTTPException(status_code=400, detail="Payload must contain 'type' and 'Velocity'")
    solution_type = payload["solution_type"]
    velocity = payload["velocity"]
    # Start the genetic algorithm in a separate thread
    ga_thread = threading.Thread(target=run_GA_in_background, args=(solution_type, velocity))
    ga_thread.start()
    return {"status": "ok"}

@App.get("/get_status")
async def getStatus():
    if not GA.Status:
        return {"fitness_tracker": GA.FitnessTracker}
    return {"status": "Complete.", "Airfoil": GA.best_solution}


def run_GA_in_background(solution_type, velocity):
    if solution_type == 1: num_generations,num_solutions_per_gen = 15, 60
    elif solution_type == 2: num_generations,num_solutions_per_gen = 15, 20
    elif solution_type == 3: num_generations,num_solutions_per_gen = 20, 10
    else: raise HTTPException(status_code=400, detail="Invalid solution type")

    GA.run_GA(num_generations=num_generations, 
              num_solutions_per_gen=num_solutions_per_gen, 
              Velocity=velocity, 
              solution_type=solution_type)

if __name__ == "__main__":
    # Specify the port here
    port = 8081
    # Run the FastAPI app with uvicorn
    uvicorn.run(App, host="127.0.0.1", port=port)