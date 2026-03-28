from pathlib import Path
import carbonx_wrapper
from simulation_setup_loader import build_kwargs


SETUP_FILE = Path(__file__).with_name("simulation_setup.txt")

model = carbonx_wrapper.GasReactor(
    **build_kwargs(
        SETUP_FILE,
        catalsyt_element="Ni",
        intnum= 37,
        bin_spacing=1.9,
        rtol=1e-12,
        atol= 1e-38,
        length_step = 'flex_loose',
        kernel_type="fuchs",
        wrapper_mapping_temp=None,
        temperature_history="custom",
        total_initial_concentration=1e+11,
        E_a1=0.9,
        __xqtot=2.01e-5,
        reactor_length=0.6,
        xdtube=0.0254,
        gas_initial_composition={"C2H2": 0.0045, "H2": 0.045, "N2": 1 - 0.0045- 0.045},
        dp_initial_premade=15e-9, 
        surface_kinetics_solver_activated=True,
        carb_struct_enabled=True,
        surface_kinetics_type="Multilayerd_Model",
    )
)  
_, solutions = model.run()



