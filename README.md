# Genetic Programming: Benchmarking Deep Memory Tasks

Benchmarking **Genetic Programming (DEAP)** and **NeuroEvolution (NEAT)** on deep-memory tasks inspired by Neural Turing Machine–style problems: storing and recalling sequences, navigating temporal mazes, and sequence-based classification.

## Tasks

| Task | Description |
|------|-------------|
| **Copy Task** | Store a sequence of bits, then output it in order (write phase → read phase). Variable sequence length and bit width (e.g. 8-bit). |
| **Sequence Recall** | Push values through a “maze” of corridors; recall them in order (LSTM-style depth with variable corridor length). Depths: 4, 5, 6, 15, 21. |
| **Sequence Classification** | Classify sequences based on temporal structure (variable depth/length). |
| **Iris** | Classic Iris classification (DEAP only). |
| **Tic-Tac-Toe** | Game-playing (DEAP only). |

## Repository Structure

```
├── DEAP/                    # Genetic Programming (tree-based)
│   ├── Copy Task/           # evolve.py, evolve_logical.py, evolve_modified.py, evolve_mul.py, evolve_vector.py, runner.py, generalize.py
│   ├── Sequence Recall/     # evolve*.py, runner.py, generalize.py, verify.py
│   ├── Sequence Classification/
│   ├── Iris/
│   └── tic-tac-toe/
├── NEAT/                    # NeuroEvolution of Augmenting Topologies
│   ├── Copy Task/           # evolve.py, run.py, visualize.py + config
│   ├── Sequence Recall/
│   └── Sequence Classification/
└── Plotting/                # Scripts for success-rate figures (e.g. gecco_paper.py)
```

**DEAP variants (primitive sets):**

- **std** — standard arithmetic/functional primitives  
- **log** — logical (AND, OR, etc.)  
- **mod** — modified task (e.g. delimiter encoding)  
- **mul** — multiplication-focused  
- **vec** — full vector (Copy Task only)

## Requirements

- Python 3
- [DEAP](https://github.com/DEAP/deap) (Genetic Programming)
- [neat-python](https://github.com/CodeReclaimers/neat-python) (NEAT)
- NumPy, scikit-learn
- matplotlib, graphviz (plotting and NEAT topology visualization)
- pandas, seaborn (Iris / analysis scripts)

Install with pip:

```bash
pip install -r requirements.txt
```

## Running Evolution

Each task folder has an `evolve*.py` script. Run from that folder, e.g.:

```bash
# DEAP Copy Task (standard)
cd "DEAP/Copy Task"
python evolve.py

# DEAP Sequence Recall (standard)
cd "DEAP/Sequence Recall"
python evolve.py

# NEAT Copy Task
cd "NEAT/Copy Task"
python evolve.py
```

Configuration (sequence length, bits, depth, generations, number of runs, etc.) is set at the top of each `evolve*.py` file.

## Testing Champions / Generalization

- **DEAP:** Use `runner.py` in each task folder for interactive champion evaluation (e.g. `std`, `log`, `mod`, champion index, number of tests). Use `generalize.py` (and `generalize_vector.py` for Copy Task) for generalization evaluation.
- **NEAT:** Use `run.py` in each NEAT task folder to evaluate saved networks.

## Plotting

Scripts in `Plotting/` load pickled reports from `*/reports/` and produce figures (e.g. success percentage vs generations, DEAP vs NEAT):

```bash
cd Plotting
python gecco_paper.py      # Main paper figures (copy task, sequence recall, sequence classification)
python copy_task.py        # Copy task–specific plots
python sequence_recall.py
python sequence_classification.py
```

Figures are written to paths defined in each script (e.g. `Plotting/Gecco Paper/`).

## Results

Reports and champions are saved under each task’s `reports/` and `champions/` directories (pickle and text). Regenerate plots by re-running the evolution scripts to refresh these files, then running the Plotting scripts.

## Citation

If you use this benchmark or code, please cite the associated work (e.g. GECCO paper / thesis) as appropriate.
