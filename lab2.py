import numpy as np

N = 16  
M = 2   

e1 = np.array([
     1,  1,  1,  1,
    -1, -1,  1, -1,
    -1,  1, -1, -1,
     1,  1,  1,  1
]) # Образ "Z"

e2 = np.array([
     1, -1, -1,  1,
    -1,  1,  1, -1,
    -1,  1,  1, -1,
     1, -1, -1,  1
]) # Образ "X"

E = np.array([e1, e2])


W = E.T / 2 

T = N / 2
E_max = 0.1

epsilon = 1 / N

def activation_function(s, T_val):
    if s <= 0:
        return 0
    elif 0 < s <= T_val:
        return s
    else:
        return T_val

def print_pattern(vector, title):
    print(f"\n{title}")
    grid = vector.reshape(4, 4)
    for row in grid:
        line = "".join(["⬛ " if val == 1 else "⬜ " for val in row])
        print(line)

print_pattern(e1, "Еталон 1 (Z)")
print_pattern(e2, "Еталон 2 (X)")

x_input = np.copy(e1)
x_input[5] = 1
print_pattern(x_input, "Вхідний образ (з шумом)")

y1 = np.zeros(M)
for j in range(M):
    s = np.dot(x_input, W[:, j]) + T     
    y1[j] = activation_function(s, T)

y2_curr = np.copy(y1)
print(f"\nПочаткові значення нейронів (y1): {y2_curr}")

iteration = 0
while True:
    y2_next = np.zeros(M)
    for j in range(M):
        sum_others = np.sum(y2_curr) - y2_curr[j]
        s_next = y2_curr[j] - epsilon * sum_others
        y2_next[j] = activation_function(s_next, T)
    
    diff = np.linalg.norm(y2_next - y2_curr)
    if diff <= E_max:
        y2_curr = y2_next
        break
    
    y2_curr = y2_next
    iteration += 1
    print(f"Ітерація {iteration}, стан нейронів: {y2_curr}")

winner = np.argmax(y2_curr)
names = ["Z", "X"]
print(f"\nРЕЗУЛЬТАТ: Мережа розпізнала образ '{names[winner]}' (Еталон №{winner + 1})")