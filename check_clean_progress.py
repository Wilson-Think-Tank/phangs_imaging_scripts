"""
Python script for visualizing CASA cleaning progress.

It takes two arguments.
  1) The path to the CASA log file associated with the cleaning you want
     to visualize the progress for.
  2) The "imagename" argument used in (t)clean for the cleaning you want
     to visualize. Note this should not include ".image" or any other
     extensions.

Changes in the total model flux and residual extrema between each major
cycle are plotted as a function of time. These values are extracted from
the CASA log file and will not interfere with the cleaning process.

It can be run on a log file for which cleaning is still taking place.
This is the primary purpose, so that long-running (t)clean jobs can be
monitored for progress and convergence/divergence. It can also be run on
a log file where the cleaning has concluded.

Note that it will show values across different (t)clean calls but that
have the same image name argument. This is intentional as this script
was designed for monitoring progress of the PHANGS version 1 imaging
pipeline, where tclean can be called multiple times to clean a single
cube to progressively lower thresholds. This script may be useful
elsewhere, but the results should be checked carefully.
"""
import matplotlib.pyplot as plt
import numpy as np
import sys

log_path = sys.argv[1]
target_image_name = sys.argv[2]

with open(log_path, "r") as log_file:
    lines = log_file.readlines()

model_time_strs = list()
residual_time_strs = list()
total_model_fluxes = list()
max_residuals_full_image = list()
min_residuals_full_image = list()
max_residuals_in_mask = list()
min_residuals_in_mask = list()

found_first_model = False
found_first_residual = False
idx = 0
while idx < len(lines):
    # retrieve the flux and residuals from the start of the first
    # imagename-matching (t)clean call
    if not found_first_model or not found_first_residual:
        find_next_values = True
    # after first (t)clean is found
    else:
        # stop scanning log file if another imagename is encountered
        if "clean(vis=" in lines[idx] and target_image_name not in lines[idx + 2]:
            break
        # search following lines for flux and residuals if end of major cycle is
        # encountered
        find_next_values = (
            "Run" in lines[idx] and "Major" in lines[idx] and "Cycle" in lines[idx]
        )

    if find_next_values:
        found_model = False
        found_residual = False
        # search for flux and residual values for current major cycle
        while not found_model or not found_residual:
            idx += 1
            # stop if end of file reached without finding next values (cleaning
            # still running so log file isn't complete yet; just show what is
            # logged so far)
            if idx == len(lines):
                break
            if (
                "[{0}] Total Model Flux : ".format(target_image_name) in lines[idx]
                or "[{0}] Peak residual (max,min) within mask :".format(
                    target_image_name
                )
                in lines[idx]
            ):
                # extract the time stamp
                line_split_priority = lines[idx].split("INFO")
                time_str = line_split_priority[0].strip()
                if "Total" in lines[idx]:
                    model_time_strs.append(time_str)
                elif "Peak" in lines[idx]:
                    residual_time_strs.append(time_str)

                # extract the flux or residual values
                line_split_colon = lines[idx].split(":")
                if "Total" in lines[idx]:
                    total_model_fluxes.append(float(line_split_colon[-1].strip()))
                elif "Peak" in lines[idx]:
                    residuals_full_image = line_split_colon[-1]
                    residuals_in_mask = line_split_colon[-2]
                    residuals_full_image_split_comma = residuals_full_image.split(",")
                    residuals_in_mask_split_comma = residuals_in_mask.split(",")
                    max_residuals_full_image.append(
                        float(residuals_full_image_split_comma[0].split("(")[1])
                    )
                    max_residuals_in_mask.append(
                        float(residuals_in_mask_split_comma[0].split("(")[1])
                    )
                    min_residuals_full_image.append(
                        float(residuals_full_image_split_comma[1].split(")")[0])
                    )
                    min_residuals_in_mask.append(
                        float(residuals_in_mask_split_comma[1].split(")")[0])
                    )

                # mark if value found for this major cycle
                if "Total" in lines[idx]:
                    found_first_model = True
                    found_model = True
                elif "Peak" in lines[idx]:
                    found_first_residual = True
                    found_residual = True

    idx += 1

model_times = np.array(model_time_strs, dtype="datetime64")
residual_times = np.array(residual_time_strs, dtype="datetime64")

total_model_flux_deltas = np.diff(total_model_fluxes)
max_residual_full_image_deltas = np.diff(max_residuals_full_image)
min_residual_full_image_deltas = np.diff(min_residuals_full_image)
max_residual_in_mask_deltas = np.diff(max_residuals_in_mask)
min_residual_in_mask_deltas = np.diff(min_residuals_in_mask)

fig, ax = plt.subplots(5, 1, sharex=True, figsize=(9, 9))

ax[0].plot(
    model_times[1:], total_model_flux_deltas, color="C0", marker="o", linestyle="None"
)
ax[1].plot(
    residual_times[1:],
    max_residual_full_image_deltas,
    color="C1",
    marker="^",
    linestyle="None",
)
ax[2].plot(
    residual_times[1:],
    min_residual_full_image_deltas,
    color="C1",
    marker="^",
    linestyle="None",
)
ax[3].plot(
    residual_times[1:],
    max_residual_in_mask_deltas,
    color="C2",
    marker="P",
    linestyle="None",
)
ax[4].plot(
    residual_times[1:],
    min_residual_in_mask_deltas,
    color="C2",
    marker="P",
    linestyle="None",
)

fig.suptitle(target_image_name)

ax[4].set_xlabel("Date")

ax[0].set_ylabel(
    r"""$\Delta$ Total
model flux"""
)
ax[1].set_ylabel(
    r"""$\Delta$ Max.
full-image
residual"""
)
ax[2].set_ylabel(
    r"""$\Delta$ Min.
full-image
residual"""
)
ax[3].set_ylabel(
    r"""$\Delta$ Max.
in-mask
residual"""
)
ax[4].set_ylabel(
    r"""$\Delta$ Min.
in-mask
residual"""
)

fig.subplots_adjust(hspace=0.05)

for label in ax[4].get_xticklabels(which="major"):
    label.set(rotation=30, rotation_mode="anchor", horizontalalignment="right")

plt.show()
