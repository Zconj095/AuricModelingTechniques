# Feedback Loop Model    
def feedback_loop(mood, em, h, nt):
    new_mood = mood + em  
    new_em = mood - h + nt
    new_h = mood - em  
    new_nt = h + em
    return new_mood, new_em, new_h, new_nt

moods = []
vals = [np.random.rand(4) for i in range(30)]  
mood, em, h, nt = vals[0]   

for i in range(30):
    new_vals = feedback_loop(mood, em, h, nt)
    mood, em, h, nt = new_vals
    moods.append(mood)