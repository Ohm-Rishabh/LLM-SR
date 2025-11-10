# Strogatz ODE Systems Documentation

This document contains the exact equations and initial conditions for the ODE systems simulated in `simulate_ode.m` from the original Strogatz dataset repository.

## Simulation Parameters
- **Time span**: 10 time units
- **Sampling time**: 0.1
- **Number of samples**: 100
- **Number of trajectories per system**: 4

---

## 1. Bacterial Respiration (pg 288, Strogatz)

### Equations
```
dx/dt = 20 - x - (x*y)/(1 + 0.5*x²)
dy/dt = 10 - (x*y)/(1 + 0.5*x²)
```

### Initial Conditions
4 random initial conditions with:
- x₀ ~ 5 + N(0, 1)
- y₀ ~ 10 + N(0, 0.1)

---

## 2. Bar Magnets (p 286, Strogatz)

### Equations
```
dx/dt = 0.5*sin(x - y) - sin(x)
dy/dt = 0.5*sin(y - x) - sin(y)
```

### Initial Conditions
4 random initial conditions with:
- x₀ ~ Uniform(0, 2π)
- y₀ ~ Uniform(0, 2π)

---

## 3. Glider (pg 188, Strogatz)

### Equations
```
dx/dt = -0.05*x² - sin(y)
dy/dt = x - cos(y)/x
```

### Initial Conditions
4 random initial conditions with:
- x₀ ~ 5 + N(0, 1)
- y₀ ~ 0 + N(0, 0.1)

---

## 4. Lotka-Volterra

### Equations
```
dx/dt = 3*x - 2*x*y - x²
dy/dt = 2*y - x*y - y²
```

### Initial Conditions
Fixed initial conditions:
1. [x₀, y₀] = [1, 3]
2. [x₀, y₀] = [4, 1]
3. [x₀, y₀] = [8, 2]
4. [x₀, y₀] = [3, 3]

---

## 5. Predator-Prey (pg 288, Strogatz)

### Equations
```
dx/dt = x*(4 - x - y/(1 + x))
dy/dt = y*(x/(1 + x) - 0.075*y)
```

### Initial Conditions
4 random initial conditions with:
- x₀ ~ 5 + N(0, 1)
- y₀ ~ 10 + N(0, 0.1)

---

## 6. Shear Flow (p 192, Strogatz)

### Equations
```
dx/dt = cot(y)*cos(x)
dy/dt = (cos²(y) + 0.1*sin²(y))*sin(x)
```

### Initial Conditions
4 random initial conditions with:
- x₀ ~ Uniform(-π, π)
- y₀ ~ Uniform(-π/2, π/2)

---

## 7. Van der Pol (p 212, Strogatz)

### Equations
```
dx/dt = 10*(y - (1/3)*(x³ - x))
dy/dt = -1/10*x
```

### Initial Conditions
4 random initial conditions with:
- x₀ ~ Uniform(0, 1)
- y₀ ~ Uniform(0, 1)

---

## Data Format

For each system, the MATLAB script generates two datasets:
- Dataset 1: [dx/dt, x, y] for each time point
- Dataset 2: [dy/dt, x, y] for each time point

Each dataset contains 400 rows (4 initial conditions × 100 time points).
