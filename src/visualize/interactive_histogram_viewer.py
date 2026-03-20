from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backend_bases import MouseEvent

from src.config.constants import LABEL_MAP
from src.utils.custom_blosc2 import load_blosc2

# Histogram scatter colors: Ghost (red), Object (green), Glass (blue)
HIST_COLORS = {
    1: "#00a65a",  # object -> green
    2: "#1f77b4",  # glass -> blue
    3: "#d62728",  # ghost -> red
}


def _to_dhw(arr: np.ndarray) -> np.ndarray:
    """Convert (X, Y, Z) to (Z, Y, X)."""
    if arr.ndim != 3:
        raise ValueError(f"Expected 3D array, got shape {arr.shape}")
    return arr.transpose(2, 1, 0)


class InteractiveHistogramViewer:
    """Interactive viewer for 2D intensity map with histogram on click."""

    def __init__(
        self,
        voxel_xyz: np.ndarray,
        prediction_xyz: np.ndarray,
    ) -> None:
        self.voxel_xyz = voxel_xyz
        self.prediction_xyz = prediction_xyz
        self.raw_dhw = _to_dhw(voxel_xyz)
        self.prediction_dhw = _to_dhw(prediction_xyz)

        # Compute 2D intensity map (sum along Z axis)
        self.intensity_map = np.sum(voxel_xyz, axis=2).T  # (Y, X) for imshow

        self.fig = None
        self.ax_map = None
        self.ax_hist = None
        self.click_marker = None
        self.current_coord = None

    def _plot_histogram(self, x: int, y: int) -> None:
        """Plot histogram for the selected coordinate."""
        self.ax_hist.clear()

        raw_series = self.raw_dhw[:, y, x]
        prediction_series = self.prediction_dhw[:, y, x]
        time_steps = np.arange(raw_series.shape[0])

        self.ax_hist.plot(
            time_steps,
            raw_series,
            color="#555555",
            linewidth=2.0,
            zorder=1,
            label="Histogram",
        )

        non_zero_mask = prediction_series > 0
        if np.any(non_zero_mask):
            for label_value in np.unique(prediction_series[non_zero_mask]):
                label_mask = prediction_series == label_value
                self.ax_hist.scatter(
                    time_steps[label_mask],
                    raw_series[label_mask],
                    s=80,
                    color=HIST_COLORS.get(int(label_value), "#000000"),
                    marker="o",
                    edgecolors="black",
                    linewidths=1.0,
                    label=LABEL_MAP.get(int(label_value), f"class_{label_value}"),
                    zorder=3,
                    alpha=0.9,
                )

        self.ax_hist.set_xlabel("Time step (Z)")
        self.ax_hist.set_ylabel("Intensity")
        self.ax_hist.set_title(f"Histogram at (X={x}, Y={y})")
        self.ax_hist.set_facecolor("white")
        self.ax_hist.set_xlim(0, self.raw_dhw.shape[0])
        self.ax_hist.legend(loc="upper right")
        self.ax_hist.grid(True, alpha=0.3)

    def _on_click(self, event: MouseEvent) -> None:
        """Handle mouse click event."""
        if event.inaxes != self.ax_map:
            return

        x = int(round(event.xdata))
        y = int(round(event.ydata))

        max_x, max_y = self.voxel_xyz.shape[0], self.voxel_xyz.shape[1]
        if not (0 <= x < max_x and 0 <= y < max_y):
            return

        self.current_coord = (x, y)

        # Update click marker
        if self.click_marker is not None:
            self.click_marker.remove()
        self.click_marker = self.ax_map.scatter(
            x, y, s=100, color="red", marker="x", linewidths=2, zorder=10
        )

        # Update histogram
        self._plot_histogram(x, y)
        self.fig.canvas.draw_idle()

    def run(self) -> None:
        """Start the interactive viewer."""
        self.fig, (self.ax_map, self.ax_hist) = plt.subplots(
            1, 2, figsize=(14, 5), gridspec_kw={"width_ratios": [1, 1.5]}
        )

        # Plot 2D intensity map
        im = self.ax_map.imshow(
            self.intensity_map,
            aspect="auto",
            origin="upper",
            cmap="viridis",
        )
        self.ax_map.set_xlabel("X")
        self.ax_map.set_ylabel("Y")
        self.ax_map.set_title("Intensity Map (click to view histogram)")
        self.fig.colorbar(im, ax=self.ax_map, label="Sum intensity")

        # Initialize empty histogram
        self.ax_hist.set_xlabel("Time step (Z)")
        self.ax_hist.set_ylabel("Intensity")
        self.ax_hist.set_title("Click on the map to view histogram")
        self.ax_hist.set_facecolor("white")
        self.ax_hist.grid(True, alpha=0.3)

        # Connect click event
        self.fig.canvas.mpl_connect("button_press_event", self._on_click)

        plt.tight_layout()
        plt.show()


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Interactive histogram viewer with 2D intensity map"
    )
    parser.add_argument(
        "voxel",
        type=Path,
        help="Path to voxel file (.b2)",
    )
    parser.add_argument(
        "prediction",
        type=Path,
        help="Path to annotation/prediction file (.b2)",
    )
    args = parser.parse_args()

    if not args.voxel.exists():
        raise FileNotFoundError(f"Missing voxel file: {args.voxel}")
    if not args.prediction.exists():
        raise FileNotFoundError(f"Missing prediction file: {args.prediction}")

    print(f"Loading voxel file: {args.voxel}")
    voxel_xyz = load_blosc2(args.voxel)
    print(f"Loading prediction file: {args.prediction}")
    prediction_xyz = load_blosc2(args.prediction)

    if voxel_xyz.shape != prediction_xyz.shape:
        raise ValueError(
            f"Shape mismatch: voxel={voxel_xyz.shape}, prediction={prediction_xyz.shape}"
        )

    print(f"Data shape: {voxel_xyz.shape} (X, Y, Z)")
    print("Click on the intensity map to view histogram at that location.")

    viewer = InteractiveHistogramViewer(voxel_xyz, prediction_xyz)
    viewer.run()


if __name__ == "__main__":
    main()
