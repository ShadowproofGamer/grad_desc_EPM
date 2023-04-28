import numpy as np
x = np.array([4,4])*2
print(x)




gradient = lambda vec: [2*(vec[0]),2*(vec[1])]
print(gradient)
start = [4, 4]
print(start)
vector = np.array(start)
print(vector)
learn_rate = 0.1
diff = -learn_rate * np.array(gradient(vector))
print(gradient(vector))
print(np.array(gradient(vector)))
print(diff)