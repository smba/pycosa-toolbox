import itertools
from abc import ABC, abstractmethod
from typing import Sequence
import pycosa.modeling as modeling
import numpy as np
import pandas as pd
import z3

class Sampler:

    def __init__(self, bin_clauses, bin_features):
        self.clauses = bin_clauses
        self.bin_features = bin_features
        
    def sample(self, size = 10):

        solutions = []
        
        solver = z3.Solver()
        solver.add(self.clauses)
    
        solver.add(
           z3.Sum([z3.If(z3.Bool(f"b_{opt}"), 1, 0) for opt in range(len(self.bin_features))]) == 10    
        )
    
        for i in range(size):
                
            if solver.check() == z3.sat:
                model = solver.model()
    
                # construct solution
                solution = {}
                for j in range(len(self.bin_features)):
                    val = model[z3.Bool(f'b_{j}')]
                    if val is not None:
                        solution[f'b_{j}'] = val.__bool__()
                    else:
                        solution[f'b_{j}'] = False
                
                
                
                solutions.append(solution)
                # add uniqueness constraint
                literals = [z3.Bool(f"b_{z}") for z in range(len(self.bin_features))]
                solver.add(
                    z3.Or([p != model.evaluate(p, model_completion=True) for p in literals])
                )
        df = pd.DataFrame(solutions)
        return df

class DistanceBasedSampler:
    
    def __init__(self, vm: modeling.VariabilityModel):
        
        self.clauses = vm.get_binary_clauses()
        self.bin_features = vm.get_binary_features()
        
    
    def sample(self, max_size=100):
        
        n_options = len(self.bin_features)
        
        # Construct some Solplt.pcolormesh(en)ver instanes
        solvers = {}
        for d in range(n_options):
            solvers[d] = z3.Solver()
            solvers[d].add(self.clauses)
            solvers[d].add(
                z3.Sum([z3.If(z3.Bool(opt), 1, 0) for opt in range(n_options)]) == d
            )
        
        solutions = []
        for i in range(max_size):

            d = np.random.choice(np.arange(n_options), size=1, replace=True)[0]
            d = int(d)

            if solvers[d].check() == z3.sat:
                model = solvers[d].model()
    
                # construct solution
                solution = {}
                for j in range(n_options):
                    option = self.bin_features[j]
                    val = model[z3.Bool(option)]
                    if val is not None:
                        solution[option] = val.__bool__()
                    else:
                        solution[option] = False
                
                solutions.append(solution)
                
                # add uniqueness constraint
                literals = [z3.Bool(option) for option in self.bin_features]
                solvers[d].add(
                    z3.Or([p != model.evaluate(p, model_completion=True) for p in literals])
                )
            
        df = pd.DataFrame(solutions)
        return df
    
    
class GroupSampler:
    
    def __init__(self, vm: modeling.VariabilityModel):
        self.vm = vm
        self.bin_features = vm.get_binary_features()
    
    def sample(self, options, max_size):
        old_solutions = []
        
        enabled = []
        disabled = []

        n_options = len(self.bin_features)

        available_distances = set(list(range(n_options)))
        
        while len(enabled) < max_size and len(available_distances) > 0:
        #for i in range(max_size):
            solver = z3.Solver()
             
            solver.add(old_solutions)
            
            self.clauses = self.vm.get_z3_pair_clauses()
            solver.add(self.clauses)
            
            
            # add distance constraint
            d = int(np.random.choice(list(available_distances), size=1)[0])
            solver.add(
                z3.Sum([z3.If(z3.Bool(f"{option}-1"), 1, 0) for option in self.bin_features]) == d   
            )

            
            for option in self.bin_features:
                if option in options:

                    solver.add(
                        z3.And(
                            z3.Bool(f"{option}-1"),
                            z3.Not(z3.Bool(f"{option}-2"))
                        )
                    )
                else:
                    solver.add(
                        z3.Bool(f"{option}-1") == z3.Bool(f"{option}-2")
                    )
            
            if solver.check() == z3.sat:
                model = solver.model()

                # construct solutions
                en_solution = {}
                dis_solution = {}
                for option in self.bin_features:
                    en_val = model[z3.Bool(f'{option}-1')]
                    dis_val = model[z3.Bool(f'{option}-2')]
                    
                    if en_val is not None and dis_val is not None:
                        en_solution[f'{option}'] = en_val.__bool__()
                        dis_solution[f'{option}'] = dis_val.__bool__()
                    else:
                        en_solution[f'{option}'] = False
                        dis_solution[f'{option}'] = False
                
                enabled.append(en_solution)
                disabled.append(dis_solution)
            
                # add uniqueness constraint
                
                en_literals = [z3.Bool(f"{option}-1") for option in self.bin_features]
                dis_literals = [z3.Bool(f"{option}-2") for option in self.bin_features]
                
                old_solutions.append(
                    z3.Or([p != model.evaluate(p, model_completion=True) for p in en_literals])
                )
                old_solutions.append(
                    z3.Or([p != model.evaluate(p, model_completion=True) for p in dis_literals])
                )
                
            else:
                #print(f"{d} is not satisfiable")
                available_distances = available_distances - set([d])
            
        enabled = pd.DataFrame(enabled)
        disabled = pd.DataFrame(disabled)
        
        return enabled, disabled
            
            
            
if __name__ == "__main__":
    pass
