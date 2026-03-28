from pathlib import Path

import carbonx_wrapper
import mapping_wrapper
import object_converter
from simulation_setup_loader import build_kwargs


SETUP_FILE = Path(__file__).with_name("simulation_setup.txt")

model = carbonx_wrapper.GasReactor(
    **build_kwargs(
        SETUP_FILE,
        catalsyt_element="Fe",
        intnum= 37,
        bin_spacing=1.55,
        rtol=1e-12,
        atol= 1e-38,
        length_step = 'flex_loose',
        kernel_type="fuchs",
        wrapper_mapping_temp=None,
        temperature_history="celnik_2008",
        total_initial_concentration=1.0749732238486255e+18,
        __xqtot=2.01e-5,
        reactor_length=0.73, #assures the equivalent residence time is ~7 sec
        xdtube=0.0254,
        gas_initial_composition={"O2": 0, "Ar": 0, "C2H2": 1e-15, "H2O": 0, "N2": 1 - 1e-15},
        dp_initial_premade=5.43e-10, # 2.52 (Fe1), 3.78 (Fe2), 5.6 (Fe4), 6.81 (Fe7)
        surface_kinetics_solver_activated=False,
        carb_struct_enabled=False,
    )
)  
_, solutions = model.run()
# Optional:
# Uncomment this line only when you want to export the model data.
# This creates a "data" folder in the current working directory.
object_converter.object_converter(model)
