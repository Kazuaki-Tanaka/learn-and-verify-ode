# Learn and Verify ODE

**A MATLAB Package for Rigorous Enclosure of ODE Solutions using Neural Networks**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![MATLAB](https://img.shields.io/badge/MATLAB-R2025b-blue.svg)](https://www.mathworks.com/products/matlab.html)

This repository provides the code for the paper:

> **"Learn and Verify: A Rigorous Approach to Enclosing Solutions of Differential Equations through Deep Learning with Sub- and Super-solutions"**  
> Kazuaki Tanaka and Kohei Yatabe

## 📖 Overview

Traditional Physics-Informed Neural Networks (PINNs) can approximate solutions to differential equations, but they lack rigorous error bounds. Our **Learn and Verify** approach addresses this limitation by:

1. **Learn** sub- and super-solutions using neural networks
2. **Verify** the learned candidates rigorously using interval arithmetic (INTLAB), guaranteeing that the true solution lies between the verified sub- and super-solutions

### Key Features

- ✅ **Mathematically rigorous** solution enclosure with guaranteed error bounds
- 🧠 **Neural network-based** learning of sub- and super-solutions
- 🔬 **Interval arithmetic** verification using INTLAB
- ⚡ **Flexible architecture** - applicable to various ODEs (currently supports scalar ODEs only)

## 🚀 Quick Start

### Prerequisites

- **MATLAB** R2025b
- **INTLAB** 14 ([download here](http://www.ti3.tu-harburg.de/rump/intlab/))
- **Deep Learning Toolbox** (for neural network functionality)

> **Note**: Operation is not guaranteed on versions other than those listed above. However, this package may work on other MATLAB versions where Deep Learning Toolbox and INTLAB are supported.

### Installation

1. Clone this repository:
```bash
git clone https://github.com/Kazuaki-Tanaka/learn-and-verify-ode.git
cd learn-and-verify-ode
```

2. Run the setup script in MATLAB to add the package to your path:
```matlab
cd learn-and-verify-ode
setup
```

3. Verify INTLAB is installed:
```matlab
intvalinit('DisplayInfsup')
```

### Basic Usage

See the [`examples/`](examples/) folder for usage examples:

- [`simple_logistic.m`](examples/simple_logistic.m) - Classical logistic equation
- [`general_logistic.m`](examples/general_logistic.m) - Logistic equation with time-varying coefficients

```matlab
% Run an example
cd examples
run simple_logistic.m
```

## 📂 Repository Structure

```
learn-and-verify-ode/
├── +OneDim/                    # Main package (MATLAB package)
│   ├── NN.m                    # Neural network model
│   ├── RigorousNN.m            # Rigorous verification with INTLAB
│   ├── IVPsolver.m             # Approximate solution solver
│   ├── IVPsolver2.m            # Sub-/super-solution solver
│   └── Optimizer.m             # ADAM optimizer
│
└── examples/                   # Usage examples
    ├── simple_logistic.m       # Classical logistic equation
    └── general_logistic.m      # Time-varying coefficients
```

## 🧩 Main Components

### 1. Neural Network Model (`NN.m`)

- Fully-connected network with customizable architecture
- SIREN initialization for sinusoidal activation functions
- Supports up to 4th-order automatic differentiation
- Hard constraint support for initial conditions

### 2. Rigorous Verification (`RigorousNN.m`)

- Interval arithmetic-based verification
- Adaptive subdivision for tight enclosures
- Returns detailed diagnostics

### 3. Solvers

- **`IVPsolver.m`**: Learn approximate solutions with penalty regularization
- **`IVPsolver2.m`**: Learn sub- and super-solutions simultaneously

### 4. Optimizer (`Optimizer.m`)

- ADAM-based optimization
- Sampling modes: random, grid, region, auto (auto switches from random to grid)

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

This research was supported by JST Fusion Oriented Research for disruptive Science and Technology (FOREST) Program (Grant Number JPMJFR202S).

## 📧 Contact

For questions or issues:
- **Kazuaki Tanaka**: tanaka@ims.sci.waseda.ac.jp
- **GitHub Issues**: [https://github.com/Kazuaki-Tanaka/learn-and-verify-ode/issues](https://github.com/Kazuaki-Tanaka/learn-and-verify-ode/issues)

---

**Keywords**: Differential Equations, Neural Networks, Deep Learning, Rigorous enclosure, Interval Arithmetic, Sub- and Super-solutions, Verified numerical computation


