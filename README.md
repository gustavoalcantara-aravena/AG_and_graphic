# 🧬 Adaptive Genetic Algorithm with Niching & Memetic Local Search

An **mono-objective evolutionary algorithm** that evolves binary chromosomes using adaptive mutation, memetic refinement, and diversity-preserving strategies.

---

## 🚀 Features

- ✅ **Adaptive mutation rate** based on convergence stagnation  
- 🌱 **Niching & Speciation** using Hamming distance clustering  
- 🧠 **Memetic local search**: bit-flip hill climbing on elite individuals  
- 🔁 **Partial population restart** to escape local optima  
- 🎯 **Tournament and roulette selection**  
- 🔧 **Uniform and single-point crossover** with adaptive control  
- 🧬 **Elite memory archive** for preserving top solutions  
- 📊 **Advanced visualization** with `matplotlib` and optional `seaborn`

---

## 🔍 Objective Function

Maximize **consecutive groups of 1s** in a binary chromosome.  
Fitness is computed as the **sum of squares of each group of 1s**, rewarding continuity over quantity.

### Example:

Chromosome: [1, 1, 1, 0, 1, 1]
Fitness = 3² + 2² = 9 + 4 = 13


---

## 🧪 How It Works

This is not a basic GA — it's a hybrid, adaptive framework inspired by:

- 🧠 **Memetic algorithms**  
- 🧩 **Niching genetic algorithms (crowding, speciation)**  
- 🔁 **Stagnation-based partial restarts**  
- 💾 **Elitist memory of top individuals**

It balances **exploration and exploitation** using adaptive rates and clustering to preserve diversity and avoid premature convergence.

---

## 📈 Output Files

- `evolucion_ag.png`: Classic fitness plot (best, average, worst)
- `resultados_ag_YYYYMMDD_HHMMSS.png`: Comprehensive 4-panel chart:
  - Fitness evolution
  - Convergence & diversity analysis
  - Fitness distribution
  - Smoothed improvement rate

---

## ⚙️ Installation

**Requirements**:
- Python 3.6+
- Optional: `numpy`, `matplotlib`, `seaborn`

---

### Install dependencies:

```bash
pip install numpy matplotlib seaborn

If numpy or seaborn are missing, the code will still run with reduced features.

```
---

## ▶️ Quick Start

python3 ag_memetico.py


You can modify the evolutionary parameters here:

if __name__ == "__main__":
    ag = AlgoritmoGenetico(
        tamanio_poblacion=200,
        longitud_cromosoma=30,
        tasa_mutacion=0.05,
        tasa_mutacion_max=0.3
    )

---

## 🎓 Academic Use

This implementation is suitable for:

Testing evolutionary operators (niching, adaptive rates, hybridization)

Research in combinatorial optimization and NP-Hard problems

Teaching advanced topics in genetic algorithms and metaheuristics

Feel free to fork and extend for use in scientific projects or coursework.

---

## 🤝 Contributing
Pull requests, ideas, and forks are welcome.
If you use this in a publication, a citation is appreciated!

---

##📄 License

MIT License

---

Author: Gustavo Alcántara, PhD Student
