# Re-defining the given values after code execution state reset
Q = 10  # Heat to be dissipated in Watts
h = 10  # Heat transfer coefficient in W/m^2K for natural convection

# Converted surface area from mm^2 to m^2
A_new = 13500 * 10**-6  # Surface area in m^2

# Calculating the new temperature difference Delta T
delta_T_new = Q / (h * A_new)
print(delta_T_new)
