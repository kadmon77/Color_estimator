import numpy as np

# Reference colors and their transformations
REFERENCE_COLORS = {
    "red": np.array([255, 0, 0]),
    "green": np.array([0, 255, 0]),
    "blue": np.array([0, 0, 255]),
    "yellow": np.array([255, 255, 0]),
    "cyan": np.array([0, 255, 255]),
    "violet": np.array([174, 138, 255]),
    "orange": np.array([255, 128, 0]),
    "pink": np.array([255, 102, 204]),
    "light_blue": np.array([135, 206, 250]),
    "light_green": np.array([144, 238, 144]),
    "brown": np.array([139, 69, 19]),
    "gray": np.array([128, 128, 128]),
}

TRANSFORMATIONS = {
    "protanopia": {
        "red": np.array([134, 122, 57]),
        "green": np.array([246, 218, 0]),
        "blue": np.array([0, 74, 156]),
        "yellow": np.array([255, 226, 90]),
        "cyan": np.array([123, 226, 203]),
        "violet": np.array([120, 156, 254]),
        "orange": np.array([184, 164, 22]),
        "pink": np.array([129, 157, 246]),
        "light_blue": np.array([208, 206, 217]),
        "light_green": np.array([235, 215, 135]),
        "brown": np.array([100, 89, 24]),
        "gray": np.array([130, 127, 128]),
    },
    "deuteranopia": {
        "red": np.array([153, 115, 0]),
        "green": np.array([255, 211, 143]),
        "blue": np.array([0, 80, 132]),
        "yellow": np.array([255, 217, 112]),
        "cyan": np.array([102, 217, 195]),
        "violet": np.array([109, 159, 250]),
        "orange": np.array([207, 155, 0]),
        "pink": np.array([158, 157, 195]),
        "light_blue": np.array([216, 201, 230]),
        "light_green": np.array([255, 207, 155]),
        "brown": np.array([112, 84, 8]),
        "gray": np.array([140, 124, 129]),
    },
    "tritanopia": {
        "red": np.array([237, 30, 25]),
        "green": np.array([113, 236, 255]),
        "blue": np.array([0, 86, 89]),
        "yellow": np.array([255, 236, 143]),
        "cyan": np.array([102, 226, 255]),
        "violet": np.array([157, 158, 171]),
        "orange": np.array([255, 122, 129]),
        "pink": np.array([248, 120, 129]),
        "light_blue": np.array([146, 219, 237]),
        "light_green": np.array([164, 226, 244]),
        "brown": np.array([141, 64, 69]),
        "gray": np.array([129, 127, 137]),
    }
}

def interpolate_color(color, daltonism_type):
    distances = {}
    for ref_color_name, ref_color in REFERENCE_COLORS.items():
        distances[ref_color_name] = np.linalg.norm(color - ref_color)

    # Find the two closest reference colors
    closest_colors = sorted(distances, key=distances.get)[:2]

    c1, c2 = closest_colors
    d1, d2 = distances[c1], distances[c2]

    # Retrieve transformations for the two closest colors
    t1 = TRANSFORMATIONS[daltonism_type][c1]
    t2 = TRANSFORMATIONS[daltonism_type][c2]

    # Weighted interpolation
    weight1 = 1 - (d1 / (d1 + d2))
    weight2 = 1 - (d2 / (d1 + d2))
    interpolated_color = t1 * weight1 + t2 * weight2

    return np.clip(interpolated_color, 0, 255).astype(int)

def simulate_color(rgb, daltonism_type):
    return interpolate_color(np.array(rgb), daltonism_type)
