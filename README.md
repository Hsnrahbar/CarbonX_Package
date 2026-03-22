<img src="./Data/logo/logo_Cx.jpg" alt="CarbonX logo" width="120px">

# CarbonX

**CarbonX** is an object-oriented Python package for simulating gas-phase synthesis of single- and multi-walled carbon nanotubes (CNTs) and metallic nanoparticles (Fe, Ni, Co) in floating-catalyst chemical vapor deposition (FCCVD) reactors. Its modular, extensible architecture couples four fully-integrated submodules: chemical kinetics, surface kinetics, particle dynamics, and CNT dynamics — along with a built-in machine learning classification module for parametric map analysis.

> 📄 Full documentation and case studies are available at [github.com/Hsnrahbar/CarbonX_Package](https://github.com/Hsnrahbar/CarbonX_Package)

---

## Table of Contents

1. [Introduction](#introduction)
2. [What Can Be Simulated?](#what-can-be-simulated)
3. [Installation](#installation)
4. [Quickstart Guide](#quickstart-guide)
5. [Input & Parameters](#input--parameters)
6. [Outputs & Post-Processing](#outputs--post-processing)
7. [License](#license)
8. [How to Cite](#how-to-cite)

---

## Introduction

Carbon nanotubes exhibit exceptional electrical, optical, and mechanical properties, making them critical candidates for applications in energy storage, sensing, and advanced composites. While CVD offers advantages in scalability and cost-effectiveness, controlling CNT morphology in FCCVD reactors remains challenging due to complex interactions between chemical kinetics, catalyst nanoparticle evolution, and carbon deposition mechanisms.

**CarbonX** addresses this challenge through a modular simulation framework integrating four fully coupled submodules:

- **Chemical kinetics** — gas-phase reaction modelling via [Cantera](https://cantera.org/), supporting mechanisms such as FFCM-2, Caltech, ABF, or user-defined YAML files.
- **Surface kinetics** — catalyst activation, deactivation, carburization, and hydrogenation (multilayered and dual-dissociation models, plus user-defined kinetics).
- **Particle dynamics** — nanoparticle evolution (inception, surface growth, coagulation, sintering) via sectional population balance models.
- **CNT dynamics** — prediction of nanotube length, diameter, wall number, and graphene layer area.

A defining feature of CarbonX is its **extensible design**: users can plug in custom surface or gas kinetics modules without modifying the core simulation engine.

---

## What Can Be Simulated?

The current version of CarbonX includes a solver for **1D plug-flow reactors** and supports:

**Physical phenomena:**
- Agglomeration, surface growth, and sintering of metallic (Ni, Fe, Co) nanoparticles
- Inception of Fe from iron pentacarbonyl (Fe(CO)₅)
- Pyrolysis of hydrocarbon feedstocks (C₂H₂, C₂H₄, CH₄, etc.)
- Surface kinetics for C₂H₂ on Fe nanoparticles (adsorption, dissociation, desorption, carburization, deactivation)
- Custom hydrocarbon–nanoparticle surface kinetics via user-defined YAML mechanisms
- Formation of graphene layers on nanoparticle surfaces
- Formation of SWCNTs and MWCNTs

**Post-processing outputs:**
- Elemental mass balance profiles
- Size distribution profiles (nanoparticles and CNTs)
- Non-dimensionalized self-preserving size distributions
- 2D parametric maps of CNT diameter, length, and wall number
- Process parameters exported to `.csv` (density, viscosity, species concentration, carbon impurity)

---

## Installation

CarbonX is available on [PyPI](https://pypi.org/project/CarbonX). Install it with:

```bash
pip install carbonx
```

All required dependencies (including Cantera) are resolved automatically. Users working in an IDE can also install it through the built-in package manager.

---

## Quickstart Guide

### Single Reactor Simulation

Use the `GasReactor` class for a single set of reactor conditions:

```python
from pathlib import Path
from carbonx import GasReactor
from carbonx.modules.simulation_setup_loader import build_kwargs

SETUP_FILE = Path("simulation_setup.txt")

model = GasReactor(
    **build_kwargs(
        SETUP_FILE,
        catalyst_element="Fe",
        bin_spacing=1.9,
        length_step="flex_tight",
        kernel_type="fuchs",
        temperature_history="celnik_2008",
        E_a1=0.9,
        reactor_length=0.52,
        xdtube=0.02,
        gas_initial_composition={"C2H2": 0.0045, "N2": 0.9},
        dp_initial_premade=5.6e-9,
        surface_kinetics_solver_activated=True,
        carb_struct_enabled=True,
        surface_kinetics_type="Surface_Kinetics_General_UDF",
    )
)

times, solutions = model.run()
```

### Parametric Map Generation

Use `MappingWrapper` to sweep over multiple reactor conditions and generate 2D parametric maps:

```python
import carbonx.modules.mapping_wrapper as mapping_wrapper

wrapper = mapping_wrapper.MappingWrapper(
    **build_kwargs(
        SETUP_FILE,
        map_param="T&P",
        # ... additional parameters
    )
)

wrapper.run()
```

The built-in ML classification module will automatically identify SWCNT and MWCNT formation regions across the generated map.

---

## Input & Parameters

CarbonX is configured via a `simulation_setup.txt` file. Key parameter categories include:

| Category | Example Parameters |
|---|---|
| Reactor geometry | `reactor_length`, `xdtube`, `temperature_history` |
| Gas chemistry | `gas_initial_composition`, `gas_mechanism` |
| Particle dynamics | `bin_spacing`, `kernel_type`, `dp_initial_premade` |
| Surface kinetics | `surface_kinetics_type`, `E_a1`, `carb_struct_enabled` |
| Parametric mapping | `map_param`, `T_range_min/max`, `P_range_min/max` |
| ML classification | `ml_lambda_`, `ml_iterations`, `ml_alpha`, `ml_post_cond` |

Refer to the [User Manual](https://github.com/Hsnrahbar/CarbonX_Package) for a complete parameter reference.

---

## Outputs & Post-Processing

After running a simulation, the `model` instance provides full access to all simulation data. Key output attributes include:

- `AA12` — gas composition, viscosity, density, and dominant hydrocarbon species
- `dp_saved` — primary particle diameter at each length step
- `D_cnt_saved` — CNT mobility diameter across bins and steps
- `wall_number_section_saved` — CNT wall number across bins and steps
- `carbon_data_saved` — full carbon population data at each step
- `y` — nanoparticle state vector (N, V, A, carbon, precursor concentration)

### Built-in Post-Processing

```python
import Results_Processor_5

times, solutions = model.solve()

AA = Results_Processor_5.ResultsPostProcessor(model)

# ψ–η diagram with experimental comparison
AA.plot_psi_eta_diagram([0.004, 0.4, 1], add_experimental=True, regime_type='CR')

# Carbon mass balance
fig, axes, data = AA.mass_balance_check()

# Relative error
fig, ax, data = AA.plot_relative_error()
```

---

## License

CarbonX is licensed under the **Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0)** license. See the `LICENSE` file for details.

---

## How to Cite

If you use CarbonX in your research, please cite the associated publication (details in the repository). Case studies and validation examples benchmarked against experimental measurements are available at:

🔗 [https://github.com/Hsnrahbar/CarbonX_Package](https://github.com/Hsnrahbar/CarbonX_Package)

---

*Developed by Hossein Rahbar.*
