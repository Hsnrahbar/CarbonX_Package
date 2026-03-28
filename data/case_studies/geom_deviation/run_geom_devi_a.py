## Note:The ##total_initial_concentration number has been decreased to 3e+15 to better observe the plateau behavior.


from pathlib import Path

import carbonx_wrapper
import object_converter
from simulation_setup_loader import build_kwargs


SETUP_FILE = Path(__file__).with_name("simulation_setup.txt")

model = carbonx_wrapper.GasReactor(
    **build_kwargs(
        SETUP_FILE,
        intnum= 27,
        bin_spacing=2.2,
        rtol=1e-12,
        atol= 1e-38,
        length_step = 'flex_extra_tight',
        kernel_type="fuchs",
        wrapper_mapping_temp=None,
        temperature_history="custom",
        total_initial_concentration=3e+15,
        E_a1=0.9,
        __xqtot=2.01e-5,
        reactor_length=0.8,
        xdtube=0.0254,
        gas_initial_composition={"O2": 0, "Ar": 0, "C2H2": 1e-15, "H2O": 0, "N2": 1 - 1e-15},
        dp_initial_premade=1000e-9,#5.6e-10,
        bundle_number=1,
        time_increment_Fe=1e-4,
        surface_kinetics_solver_activated=False,
        carb_struct_enabled=False,
        surface_kinetics_type="Multilayerd_Model",
    )
)  # "Surface_Kinetics_Ma_etal_2005" "Surface_Kinetics_Puretzky_etal_2005"
_, solutions = model.run()
# Optional:
# Uncomment this line only when you want to export the model data.
# This creates a "data" folder in the current working directory.
#object_converter.object_converter(model)


import Results_Processor 
AA=Results_Processor.ResultsPostProcessor(model) 
fig, ax, sigma_g_g, sigma_g_m, sigma_g_v=AA.plot_geometric_standard_deviations(plot_until=5000, figsize=(10, 6))

