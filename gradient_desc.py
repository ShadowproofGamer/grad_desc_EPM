import numpy as np


def gradient_descent(

    gradient, start, learn_rate, n_iter=50, tolerance=1e-03, r=1, r_iter=1

):

    vector = start

    for _ in range(n_iter):

        diff = -learn_rate * gradient(*vector)

        if np.all(np.abs(diff) <= tolerance):

            break

        vector += diff
        r*=r_iter

    return vector

def gradient_descent_2d(

    gradient, start, learn_rate, n_iter=50, tolerance=1e-03, r_iter=2, r=1

):
    k=1
    x, y = start

    for _ in range(n_iter):
        
        temp_grad = gradient(x, y, r)
        diff = [-learn_rate * temp_grad[0], -learn_rate * temp_grad[1]]
        
        stop = ((diff[0])**2 + (diff[1])**2)**(1/2)
        #if np.all(np.abs(diff) <= tolerance): break
        if stop <= tolerance: break

        x+= diff[0]
        y+= diff[1]
        k+= 1
        #learn_rate*=0.98
        #r*=r_iter

    print("iteracja: {}, warunek stop: {}<{}, result: {}, {}".format(k, stop, tolerance, x, y))
    return x, y


#zad2 gradient
#grad2=lambda x1, x2: [x2*(x2-1)*(2*x2-1)+4*r*(x1**2 -2*x1+x2**2)*(x1-1), x2*(x2-1)*(2*x2-1)+4*r*(x1**2 -2*x1+x2**2)*(x2)]
def grad2(x1, x2, r=0.05):
    if((x1**2 -2*x1 + x2**2)>0):
        return [x2*(x2-1)*(2*x2-1)+4*r*(x1**2 -2*x1+x2**2)*(x1-1), x2*(x2-1)*(2*x2-1)+4*r*(x1**2 -2*x1+x2**2)*(x2)]
    else:
        return [x2*(x2-1)*(2*x2-1), x2*(x2-1)*(2*x2-1)]

def orig_func2(vector):
    x1, x2 = vector
    print("wynik: ", (x1-1)*(x2-1)*x1*x2)

orig_func2(gradient_descent_2d(grad2, (2, 2), 0.1, 5000, 0.0001, 1.05, 0.1))


def grad3(x1, x2, r=0.05, r_iter=1):
    r*=r_iter
    if((x1**2 -2*x1 + x2**2)>0):
        return [x2*(x2-1)*(2*x2-1)+4*r*(x1**2 -2*x1+x2**2)*(x1-1), x2*(x2-1)*(2*x2-1)+4*r*(x1**2 -2*x1+x2**2)*(x2)]
    else:
        return [x2*(x2-1)*(2*x2-1), x2*(x2-1)*(2*x2-1)]

def orig_func3(vector):
    x1, x2 = vector
    print("wynik: ", (x1-1)*(x2-1)*x1*x2)