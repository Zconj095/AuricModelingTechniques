import numpy as np 

# Neurotransmitter Interaction Model
def neurotransmitter_model(nts, auric_state):
    return sum(nts) * auric_state

time = np.linspace(0, 10, 30)

nts = [np.random.rand(30) for i in range(3)] # Make nts length 30
auric_state = np.linspace(0, 1, 30)  

mood = neurotransmitter_model(nts, auric_state) 