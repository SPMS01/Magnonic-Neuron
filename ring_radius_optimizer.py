import os
import shutil
import textwrap
import utils
from magnetisation_plotter import plot_magnetisation
import numpy as np
import objective_function
import time

OUTPUT_DIR = "ring_radius_optimization_results"

shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
os.mkdir(OUTPUT_DIR)

results = []

for ring_width in range(100, 301, 10): # in nm
    for ring_radius in range(200, 501, 10): # in nm
        t = time.time()
        template = f"""
        SetCellSize(20e-9, 5e-9, 50e-9)
        SetGridSize(250, {42 + (ring_radius*2)/5}, 1)

        B_ext = Vector(0, 0, 300e-3)
        Msat = 0
        Aex = 0
        alpha = 0

        yig := 1
        excitation_region := 2
        dampener_min_region := 10
        dampener_max_region := 49

        Msat.SetRegion(yig, 140e3)
        Aex.SetRegion(yig, 3.5e-12)
        alpha.SetRegion(yig, 2e-4)

        input_waveguide := Cuboid(5e-6, 100e-9, 50e-9).Transl(0, {55+ring_radius}e-9, 0)
        ring_ext := Cylinder({ring_radius*2}e-9, 1)
        ring_int := Cylinder({(ring_radius - ring_width)*2}e-9, 1)
        ring := ring_ext.Sub(ring_int).Transl(0, 0, 0)
        output_waveguide := Cuboid(5e-6, 100e-9, 50e-9).Transl(0, -{55+ring_radius}e-9, 0)
        neuron := ring.add(input_waveguide).add(output_waveguide)
        SetGeom(neuron)
        DefRegion(yig, neuron)

        alpha_max := 0.5
        damping_slice_width := 20e-9
        damping_region_width := (dampener_max_region - dampener_min_region + 1) * damping_slice_width

        for i := dampener_min_region; i <= dampener_max_region; i++ {{
            offset := (i - dampener_min_region)

            input_waveguide_right_lb := 2.5e-6 - damping_region_width + offset * damping_slice_width
            input_waveguide_right_rb := input_waveguide_right_lb + damping_slice_width

            output_waveguide_right_lb := 2.5e-6 - damping_region_width + offset * damping_slice_width
            output_waveguide_right_rb := output_waveguide_right_lb + damping_slice_width

            coupler_right_lb := 4.7e-6 - damping_region_width + offset * damping_slice_width
            coupler_right_rb := coupler_right_lb + damping_slice_width
            right_dampener_region := xrange(input_waveguide_right_lb, input_waveguide_right_rb).
                intersect(input_waveguide).
                add(xrange(output_waveguide_right_lb, output_waveguide_right_rb).
                intersect(output_waveguide))

            left_rb := -2.5e-6 + damping_region_width - offset * damping_slice_width
            left_lb := left_rb - damping_slice_width

            coupler_left_rb := -2.62e-6 + damping_region_width - offset * damping_slice_width
            coupler_left_lb := coupler_left_rb - damping_slice_width
            left_dampener_region := xrange(left_lb, left_rb).
                intersect(input_waveguide).
                add(xrange(left_lb, left_rb).
                intersect(output_waveguide))

            dampener_region := right_dampener_region.add(left_dampener_region)
            DefRegion(i, dampener_region)
            alpha.SetRegion(i, alpha_max * (1 - exp(-5.0 * (i - dampener_min_region) / (dampener_max_region - dampener_min_region))))
            Msat.SetRegion(i, 140e3)
            Aex.SetRegion(i, 3.5e-12)
        }}

        Relax()
        print(m)
        DefRegion(excitation_region, xrange(1.64e-6, 1.7e-6).intersect(input_waveguide))
        alpha.SetRegion(excitation_region, 2e-4)
        Msat.SetRegion(excitation_region, 140e3)
        Aex.SetRegion(excitation_region, 3.5e-12)
        B_ext.SetRegion(excitation_region, Vector(10e-3 * sin(2 * pi * 6.7e9 * (t-3e-9)), 0, 300e-3))

        Snapshot(Msat)
        Snapshot(Aex)
        Snapshot(alpha)
        Snapshot(geom)
        Snapshot(regions)

        Save(m_full)
        AutoSave(m_full, 50e-12)

        Run(160e-9)
        """

        filename = os.path.join(OUTPUT_DIR, f"ring_width_{ring_width}nm_radius_{ring_radius}nm.mx3")
        mx3_filename = filename + ".mx3"
        mx3_output_folder = filename + ".out"

        with open(mx3_filename, "w+") as f:
            f.write(textwrap.dedent(template).strip())

        utils.run_mx3(
            mx3_exe_path='mumax3',
            mx3_exe_convert_path='mumax3-convert',
            mx3_file_path=mx3_filename,
            output_dir=OUTPUT_DIR
        )

        plot_magnetisation(dx=20e-9, 
                           dy=5e-9, 
                           file_path=os.path.join(mx3_output_folder, "m_full003000.npy"),
                           output_path=os.path.join(OUTPUT_DIR, f"ring_width_{ring_width}nm_radius_{ring_radius}nm_magnetisation.png"))
        accumulated_energy = objective_function.temporary_evaluate_objective(
            input_dir=mx3_output_folder,
            x_range=(175, 176),
            y_range=(0, 20)
        )

        results.append((ring_width, ring_radius, accumulated_energy))
        print(f"Ring width: {ring_width} nm, Ring radius: {ring_radius} nm, Accumulated energy: {accumulated_energy:.6g}, Time taken: {time.time() - t:.2f} seconds")

        shutil.rmtree(mx3_output_folder, ignore_errors=True)

with open(os.path.join(OUTPUT_DIR, "ring_radius_optimization_results.txt"), "w+") as result_file:
    result_file.write("Ring Width (nm), Ring Radius (nm), Accumulated Energy (a.u.)\n")
    for ring_width, ring_radius, energy in results:
        result_file.write(f"{ring_width}, {ring_radius}, {energy:.6g}\n")