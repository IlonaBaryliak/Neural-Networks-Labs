import numpy as np

# Функція активації hardlim
def hardlim(n):
    return (n >= 0).astype(int)

# значення ваг W (3 нейрони на 2 входи)
W = np.array([
    [0.5, -0.2],  
    [-0.1, 0.8],  
    [0.4, 0.4]    
])

#зміщення b (вектор 3x1)
b = np.array([
    [-1.0], 
    [0.5], 
    [1.0]
])

print("Параметри нейронної мережі (зафіксовані):")
print(f"Матриця ваг W:\n{W}")
print(f"Вектор зміщень b:\n{b}")

#тестові вектори p 
p_test = np.array([
    [1.5, -2.0, 0.0, 5.2, -8.1], # Перша координата (x)
    [2.3, 4.1, -1.0, -3.0, 6.5]   # Друга координата (y)
])

print("\nПеревірка роботи на фіксованих векторах:")
for i in range(5):
    p = p_test[:, i].reshape(2, 1)
    net_input = np.dot(W, p) + b
    a = hardlim(net_input)
    print(f"P{i+1}: [{p[0,0]:.2f}, {p[1,0]:.2f}] -> a: {a.flatten()}")