# Auric Sensation as Emotional Response
def emotion_model(E_a, P_a, E_b, P_b):  
    A_s = k1*E_a + k2*P_a + k3*E_b + k4*P_b 
    return A_s

E_a = 2
P_a = 0.5  
E_b = 1.5 
P_b = 0.8  
k1, k2, k3, k4 = 0.2, 0.3, 0.4, 0.5

A_s = emotion_model(E_a, P_a, E_b, P_b)
print(f"Auric Sensation (Emotion): {A_s}")
