LABEL_MAP: dict[int, str] = {0: "noise", 1: "object", 2: "glass", 3: "ghost"}
# Define colors for each class (excluding noise class 0)
CLASS_COLORS = {
    0: "#808080",  # gray for noise
    1: "#1f77b4",  # blue for object
    2: "#ff7f0e",  # orange for glass
    3: "#2ca02c",  # green for ghost
}
ORIGINAL_HISTOGRAM_BINS: int = 700
ORIGINAL_WIDTH: int = 400
ORIGINAL_HEIGHT: int = 512
