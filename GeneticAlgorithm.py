import joblib
import pandas as pd
import requests
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import random
import warnings
import time
warnings.filterwarnings("ignore")    

CL_Predictor = joblib.load("CL_model.pkl")
CD_Predictor = joblib.load("CD_model.pkl")
preprocess_pipeline = joblib.load('pipeline.pkl')


class Airfoil:
    def __init__(self, Velocity, AOA, d2Yl, y_TE, a_TE):
        self.Velocity = Velocity
        self.AOA = AOA
        self.Inaccuracy = 0
        self.d2Yl = d2Yl
        self.y_TE = y_TE
        self.a_TE = a_TE


    def scale_self(self):
        columns = ['Velocity', 'AOA', 'Inaccuracy', 'd2Yl', 'y_TE', 'a_TE']
        data = pd.DataFrame([[self.Velocity, self.AOA, self.Inaccuracy, self.d2Yl, self.y_TE, self.a_TE]], columns=columns)
        transformed_data = preprocess_pipeline.transform(data)
        self.Velocity, self.AOA, self.Inaccuracy, self.d2Yl, self.y_TE, self.a_TE = transformed_data[0]


    def get_fitness(self, solution_type=1):
        input_data = [[self.Velocity, self.AOA, self.Inaccuracy, self.d2Yl, self.y_TE, self.a_TE]]
        CL = CL_Predictor.predict(input_data)[0]
        input_data_with_CL = [[self.Velocity, self.AOA, self.Inaccuracy, self.d2Yl, self.y_TE, self.a_TE, CL]]
        CD = CD_Predictor.predict(input_data_with_CL)[0]
        
        # Goal variable will indicate the problem we are attempting to solve.
        # 1. Cruise Condition (CL set, minimise Drag)
        # 2. Optimal Climb (Optimal E)
        # 3. Max Climb (Maximise CL, within reason, below a ceiling.)
        match solution_type:
            case 1:
                # 1. Cruise Condition (CL set, minimise Drag)
                cl_error = abs(CL - Cruise_CL) / Cruise_CL
                if cl_error > 0.02:
                    return 1 / (1 + cl_error)  # Penalize solutions outside the range, but provide a gradient
                return (1 / CD)
            case 2:
                # 2. Optimal Climb (Optimal E)
                if CL < Cruise_CL:
                    return 1
                return (CL / CD)
            case 3:
                # 3. Max Climb (Maximise CL, within reason, below a ceiling.)
                return CL
            
    def get_CL(self):
        input_data = [[self.Velocity, self.AOA, self.Inaccuracy, self.d2Yl, self.y_TE, self.a_TE]]
        CL = CL_Predictor.predict(input_data)[0]
        return CL
    
    def get_CD(self):
        input_data = [[self.Velocity, self.AOA, self.Inaccuracy, self.d2Yl, self.y_TE, self.a_TE]]
        CL = CL_Predictor.predict(input_data)[0]
        input_data_with_CL = [[self.Velocity, self.AOA, self.Inaccuracy, self.d2Yl, self.y_TE, self.a_TE, CL]]
        CD = CD_Predictor.predict(input_data_with_CL)[0]
        return CD
        

def create_solution():
    s = Airfoil(Velocity=V, 
                 AOA=random.uniform(-3, 9), 
                 d2Yl=random.uniform(-0.4, 0.4), 
                 y_TE=random.uniform(-0.15, 0.1), 
                 a_TE=random.uniform(-18.5, 7.5))
    s.scale_self()
    return s

def init_population(population_size):
    return [create_solution() for _ in range(population_size)]

def run_GA(num_generations, num_solutions_per_gen, Velocity, solution_type=1):
    global V
    V = Velocity
    global best_solution
    global FitnessTracker
    FitnessTracker = [()]
    global Status
    Status = False
    global Cruise_CL
    Cruise_CL = ((2*16019.73)/((V**2)* 0.95697 * 16.16)) * 1.34  # Semantic value within the context of the problem.

    start_time = time.time()
    population = init_population(num_solutions_per_gen)
    pop_size = num_solutions_per_gen

    for i in range(num_generations):
        scored_population = [(s.get_fitness(solution_type), s) for s in population]
        scored_population.sort(key=sortByScore, reverse=True)

        print("\n\n=== Gen {} === ".format(i))
        # Append the Fitness Tracker
        top_fitness = scored_population[0][0]
        FitnessTracker.append((i+1, top_fitness))
        
        # Reassign the population to the newly generated population
        population = select_pool(scored_population=scored_population)


    best_solution = scored_population[0][1]
    print("\n\n=== Best solution overall === ")
    print("Fitness: ", scored_population[0][0])
    print("CL: ", best_solution.get_CL())
    print("CD: ", best_solution.get_CD())
    best_solution = denormalize_data(best_solution)
    printSolution(best_solution)

    end_time = time.time()  # Stop the timer
    execution_time = end_time - start_time  # Calculate the execution time
    print(f"\nExecution time: {execution_time:.2f} seconds")

    plot_airfoil(best_solution)
    Status = True


def tournament_select(population, k=3):
    selected = random.sample(population, k)
    return max(selected, key=lambda x: x[0])

def crossover(p1, p2):
    dna1 = [p1.AOA, p1.d2Yl, p1.y_TE, p1.a_TE]
    dna2 = [p2.AOA, p2.d2Yl, p2.y_TE, p2.a_TE]
    
    child = [dna1[n] if random.randint(0, 1) == 0 else dna2[n] for n in range(len(dna1))]
    return Airfoil(Velocity=p1.Velocity, AOA=child[0], d2Yl=child[1], y_TE=child[2], a_TE=child[3])

def blx_alpha_crossover(p1, p2, alpha=0.5):
    dna1 = [p1.AOA, p1.d2Yl, p1.y_TE, p1.a_TE]
    dna2 = [p2.AOA, p2.d2Yl, p2.y_TE, p2.a_TE]
    
    child = []
    for g1, g2 in zip(dna1, dna2):
        min_val = min(g1, g2)
        max_val = max(g1, g2)
        range_val = max_val - min_val
        child.append(random.uniform(min_val - alpha * range_val, max_val + alpha * range_val))
    
    return Airfoil(Velocity=p1.Velocity, AOA=child[0], d2Yl=child[1], y_TE=child[2], a_TE=child[3])

def mutate(s, mutation_chance=0.2, mutation_range=0.25):
    dna = [s.AOA, s.d2Yl, s.y_TE, s.a_TE]
    mutated_dna = [
        gene * (1 + random.gauss(0, mutation_range)) 
        if random.random() <= mutation_chance else gene 
        for gene in dna
        ]
    return Airfoil(Velocity=s.Velocity, AOA=mutated_dna[0], d2Yl=mutated_dna[1], y_TE=mutated_dna[2], a_TE=mutated_dna[3])

def select_pool(scored_population):
    pop_size = len(scored_population)
    new_pop = []
    
    # Elitism: Keep the best solution
    # keep 3 copies of it, mutate 2 of them.
    new_pop.append(scored_population[0][1])
    new_pop.append(mutate(scored_population[0][1], mutation_chance=1, mutation_range=0.1))
    new_pop.append(mutate(scored_population[0][1], mutation_chance=1, mutation_range=0.1))
    
    while len(new_pop) < pop_size:
        parent1 = tournament_select(scored_population)[1]
        parent2 = tournament_select(scored_population)[1]
        child = blx_alpha_crossover(parent1, parent2)
        new_pop.append(mutate(child))
    
    return new_pop

def sortByScore(tup):
    return tup[0]

def printSolution(s):
    print("V={}, AOA={}, d2Yl={}, y_TE={}, a_TE={}".format(s.Velocity, s.AOA, s.d2Yl, s.y_TE, s.a_TE))

def denormalize_data(s):
    df = pd.read_csv("PreProcessing.csv")
    Min_Max_cols = ['Velocity','d2Yl','y_TE','a_TE']
    Standard_scaler_cols = ['AOA']
    data = [[s.Velocity, s.AOA, s.d2Yl, s.y_TE, s.a_TE]]
    data_df = pd.DataFrame(data, columns=['Velocity', 'AOA','d2Yl','y_TE','a_TE'])
    
    for col in Min_Max_cols:
        min_max_scaler = MinMaxScaler()
        min_max_scaler.fit(df[[col]])
        data_df[col] = min_max_scaler.inverse_transform(data_df[[col]])
    
    for col in Standard_scaler_cols:
        std_scaler = StandardScaler()
        std_scaler.fit(df[[col]])
        data_df[col] = std_scaler.inverse_transform(data_df[[col]])

    return Airfoil(Velocity=data_df["Velocity"][0], AOA=data_df["AOA"][0], 
               d2Yl=data_df["d2Yl"][0], y_TE=data_df["y_TE"][0], a_TE=data_df["a_TE"][0])


def plot_airfoil(s):
    url = "http://localhost:8080/plotAirfoil"
    
    # Define the JSON body
    json_body = {
        "d2Yl": s.d2Yl,
        "y_TE": s.y_TE,
        "a_TE": s.a_TE, 
        "aoa": s.AOA
    }
    
    # Send the POST request
    response = requests.post(url, json=json_body)
    
    try:
        if response.status_code == 200:
            print("POST request successful!")
            print("Response:", response.json())
        else:
            print("POST request failed with status code:", response.status_code)
    except requests.exceptions.HTTPError as errh:
        print("HTTP Error:", errh)
    except requests.exceptions.ConnectionError as errc:
        print("Error connecting to the server:", errc)
    except requests.exceptions.Timeout as errt:
        print("Timeout error:", errt)
    except requests.exceptions.RequestException as err:
        print("An unexpected error occurred:", err)
    except requests.exceptions.ConnectionRefusedError as errCR:
        print("API Server is likely shut off.", errCR)
    