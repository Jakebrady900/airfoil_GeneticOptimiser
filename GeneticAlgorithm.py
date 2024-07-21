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
                if CL < Cruise_CL * 0.95 or CL > Cruise_CL * 1.05:
                    return 1
                return (CL / CD)
            case 2:
                # 2. Optimal Climb (Optimal E)
                if CL < Cruise_CL:
                    return 1
                return (CL / CD)
            case 3:
                # 3. Max Climb (Maximise CL, within reason, below a ceiling.)
                return CL
        

def create_solution():
    s = Airfoil(Velocity=V, 
                 AOA=random.uniform(-3, 9), 
                 d2Yl=random.uniform(-0.4, 0.4), 
                 y_TE=random.uniform(-0.15, 0.1), 
                 a_TE=random.uniform(-18.5, 7.5))
    s.scale_self()
    return s

def create_cruise_solution():
    s = Airfoil(Velocity=V, 
                 AOA=random.uniform(-2, 3), 
                 d2Yl=random.uniform(-0.2, 0.2), 
                 y_TE=random.uniform(-0.1, 0.1), 
                 a_TE=random.uniform(-10, 5))
    s.scale_self()
    return s

def init_population(population_size, isCruise=False):
    if isCruise:
        return [create_cruise_solution() for _ in range(population_size)]
    return [create_solution() for _ in range(population_size)]


def crossover(p1, p2):
    dna1 = [p1.AOA, p1.d2Yl, p1.y_TE, p1.a_TE]
    dna2 = [p2.AOA, p2.d2Yl, p2.y_TE, p2.a_TE]
    
    child = [dna1[n] if random.randint(0, 1) == 0 else dna2[n] for n in range(len(dna1))]
    return Airfoil(Velocity=p1.Velocity, AOA=child[0], d2Yl=child[1], y_TE=child[2], a_TE=child[3])

def mutate(s, mutation_chance=0.01):
    dna = [s.AOA, s.d2Yl, s.y_TE, s.a_TE]

    mutated_dna = [(random.uniform(-0.5, 0.5) + 1) * gene if random.random() <= mutation_chance else gene for gene in dna]
    return Airfoil(Velocity=s.Velocity, AOA=dna[0], d2Yl=dna[1], y_TE=dna[2], a_TE=dna[3])


def run_GA(num_generations, num_solutions_per_gen, Velocity, solution_type=1):
    start_time = time.time()
    global V
    V = Velocity
    global best_solution
    global FitnessTracker
    FitnessTracker = [()]
    global Status
    Status = False
    global Cruise_CL
    Cruise_CL = (2*1633)/((V**2)* 0.95697 * 16.16 * 1.25) # Semantic value within the context of the problem.

    population = init_population(population_size=num_solutions_per_gen, isCruise=(solution_type==1))
    pop_size = num_solutions_per_gen

    for i in range(num_generations):
        scored_population = [(s.get_fitness(solution_type), s) for s in population]
        scored_population.sort(key=sortByScore, reverse=True)

        print("\n\n=== Gen {} === ".format(i))
        # Append the Fitness Tracker
        avg_fitness = sum(score for score, _ in scored_population) / num_solutions_per_gen
        avg_fitness = scored_population[0][0]
        FitnessTracker.append((i+1, avg_fitness))
        
        if scored_population[0][0] == 1:
            population = getCruiseAirfoil(num_solutions_per_gen)
            continue
        
        # Reassign the population to the newly generated population
        population = select_pool(scored_population=scored_population, solution_type=solution_type)


    best_solution = scored_population[0][1]
    best_solution = denormalize_data(best_solution)
    print("\n\n=== Best solution overall === ")
    print("Fitness: ", scored_population[0][0])
    printSolution(best_solution)

    end_time = time.time()  # Stop the timer
    execution_time = end_time - start_time  # Calculate the execution time
    print(f"\nExecution time: {execution_time:.2f} seconds")

    plot_airfoil(best_solution)
    Status = True

def select_pool(scored_population, solution_type):
    # ================ Pool Selection ================
    # 1. retain the top 25% of the previous population
    # 2. crossover (returning two offspring) for 50%
    # 3. completely random new solutions for 25%
    # ====================   End   ====================

    pop_size = len(scored_population)

    if solution_type == 1 and scored_population[0][0] != 1:
        # Find how many solutions are not equal to 1
        count = 0
        for i in range(pop_size):
            if scored_population[i][0] > 1:
                count += 1
            else:
                exit
        if count < pop_size//2:
            # add the solutions that are not equal to 1 to an array on repeat until the population is full
            new_pop = [scored_population[i][1] for i in range(count)]

            # new code to fill the rest of new_pop
            while len(new_pop) < pop_size:
                new_pop.extend(scored_population[i][1] for i in range(count))

            # if new_pop is larger than pop_size, trim it down
            if len(new_pop) > pop_size:
                new_pop = new_pop[:pop_size]

    # 1.retain the top 25% of the previous population
    top_parents  = [s for _, s in scored_population[:pop_size//4]]
    
    # 2. crossover (returning two offspring) for 50%
    offspring = [crossover(random.choice(top_parents), random.choice(top_parents)) for _ in range(pop_size//2)]

    # 3. completely random new solutions for 25%
    new_solutions = [create_solution() for _ in range(pop_size - len(top_parents) - len(offspring))]

    
    # Mutation
    new_pop = top_parents + offspring + new_solutions
    new_pop = [mutate(new_pop[i]) for i in range(pop_size)]

    return new_pop


def getCruiseAirfoil(pop_size):
    return_pop = []
    target = pop_size//10
    print("Regenerating {} solutions for Cruise condtions.".format(target))
    i=0
    while i < target:
        s = create_cruise_solution()
        if s.get_fitness(1) > 1:
            return_pop.append(s)
            i += 1
            print(i)
    while len(return_pop) < pop_size:
        return_pop.append(create_cruise_solution())
    return return_pop

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
    