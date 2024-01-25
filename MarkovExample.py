# State Space Model
# Simple Markov model example

activ_probs = [0.6, 0.4, 
                0.2, 0.7]
                
intens_probs = [0.5, 0.3,
                0.8, 0.1]
                
state_probs = [[0.3, 0.1], 
               [0.2, 0.4]]
               
still_state = 1              

A_s = state_probs[still_state][still_state] 
print(f"Auric Stillness Probability: {A_s}")
