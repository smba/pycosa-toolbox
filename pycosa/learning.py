import pycosa.modeling as modeling
import pycosa.sampling as modeling

import numpy as np

class GroupLearner:
    
    def __init__(self, options):
        self.t1 = 0.5
        self.t2 = 0.1
        
        self.options = set(options)
        
        self.records = {
            option: np.array([0,1]) for option in options    
        }
        
        
    
    def _classify(self, group, perf_en, perf_dis):
        
        
        
        n_pairs = len(perf_en)
        
        delta = np.abs(perf_en - perf_dis)
        mins = [min(abs(perf_en[i]), abs(perf_dis[i])) for i in range(n_pairs)]
        

        leq_t1 = [
            delta[i] <= self.t1 * min(
                np.abs(perf_en[i]),
                np.abs(perf_dis[i])
            ) or delta[i] < 3 for i in range(n_pairs) 
        ]
        
        g_t1 = [
            not(g) for g in leq_t1
        ]
        
        mean_delta = np.mean(delta)
        g_t2 = [
            np.abs(d - mean_delta) > self.t2 * mean_delta for d in delta
        ]
        #print(np.mean(perf_en))
        if all(g_t1):
            if any(g_t2):
                c = (1,1)
            else:
                c = (1,0)
        elif any(leq_t1) and any(g_t1):
            c = (0,1)
        elif all(leq_t1):
            print("fuck da")
            self.options = self.options - set(group)
            c = (0,0)
        else:
            
            c = (0,0)
        
        # stoppage criteria
        # record covered options
        #if c == 4:
        #    print(4)
            
        
        for option in group:
            self.records[option] += np.array(c)

            
        return c
    
    def suggest_options(self, size = 5):
        
        options = np.random.choice(list(self.options), size=size)
        
        return options
if __name__ == "__main__":
    pass
