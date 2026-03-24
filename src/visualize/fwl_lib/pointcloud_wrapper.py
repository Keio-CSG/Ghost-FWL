import dataclasses
from typing import List, Optional, Union

import blosc2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks, medfilt


# Define the dataclass for DistanceResult
@dataclasses.dataclass
class DistanceResult:
    distance_m: float
    start_fwhm_bin: float
    peak_bin: int
    fwhm_bins: Optional[List[float]] = None  # [start_fwhm_bin, end_fwhm_bin]
    notes: str = ""


class PointcloudWrapper:
    def __init__(
        self,
        data: Union[pd.DataFrame, str],
        mode: str = "SPD",
        spd_mode: int = 15,
        histo_offset: int = 9,
    ) -> None:
        print("PointcloudWrapper __init__")
        self.histo_offset = histo_offset
        print(f"histo_offset: {histo_offset}")

        self.df = self.load_dataframe_or_npy(data)

        self.mode = mode.upper()
        self.spd_mode = spd_mode
        if self.mode == "SPD" and self.spd_mode not in [1, 5, 12, 15]:
            raise ValueError("Invalid SPD mode. Choose from 1, 5, 12, 15.")

        if self.mode == "SPD" and self.spd_mode == 15:
            self.histograms: np.ndarray = self._get_histogram(self.df)
            self.raw_intensity: np.ndarray = None
            self.intensity: np.ndarray = None

    def crop_xy(self, x_min: int, x_max: int, y_min: int, y_max: int) -> None:
        """
        Crop the point cloud within the specified x, y range.
        """
        df = self.df
        cropped_df = df[
            (df["//Pixel_X"] >= x_min)
            & (df["//Pixel_X"] <= x_max)
            & (df["Pixel_Y"] >= y_min)
            & (df["Pixel_Y"] <= y_max)
        ].copy()
        self.update_dataframe(cropped_df)

    def load_dataframe_or_npy(self, data_source: Union[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Determine the input type (DataFrame or npy/b2/npz path) and return a normalized DataFrame.
        - If HistData is list/ndarray, convert it to a string (parse_hist_data expects this format)
        """
        if isinstance(data_source, pd.DataFrame):
            df = data_source.copy()
        elif isinstance(data_source, str) and data_source.endswith((".npy", ".b2", ".npz")):
            if data_source.endswith(".npz"):
                npz_data = np.load(data_source, allow_pickle=True)
                if not all(k in npz_data for k in ["pixel_x", "pixel_y", "hist_data"]):
                    raise ValueError(
                        "NPZ file must contain 'pixel_x', 'pixel_y', and 'hist_data' arrays."
                    )

                df = pd.DataFrame(
                    {
                        "//Pixel_X": npz_data["pixel_x"],
                        "Pixel_Y": npz_data["pixel_y"],
                        "HistData": npz_data["hist_data"],
                    }
                )
            else:  # .npy or .b2
                if data_source.endswith(".npy"):
                    data = np.load(data_source, allow_pickle=True)
                else:  # .b2
                    with open(data_source, "rb") as f:
                        data = blosc2.unpack_array2(f.read())

                if data.ndim == 3:
                    width, height, _ = data.shape
                    xs, ys = np.meshgrid(np.arange(width), np.arange(height), indexing="ij")
                    df = pd.DataFrame(
                        {
                            "//Pixel_X": xs.flatten(),
                            "Pixel_Y": ys.flatten(),
                            "HistData": [
                                data[x, y, self.histo_offset :]
                                for x, y in zip(xs.flatten(), ys.flatten())
                            ],
                        }
                    )
                elif data.ndim == 2 and data.shape[1] == 3:
                    df = pd.DataFrame(data, columns=["//Pixel_X", "Pixel_Y", "HistData"])
                else:
                    raise ValueError(f"Expected (N, 3) or 3D data. Got shape: {data.shape}")
        else:
            raise ValueError(f"Unsupported data source: {data_source}")
        return df

    def update_dataframe(self, df: pd.DataFrame) -> None:
        """Update the internal state with a new DataFrame"""
        self.df = df
        self.histograms: np.ndarray = self._get_histogram(df)

    def get(
        self,
        method: str = "strongest",
        intensity_gain: int = 10,
        stack_num: int = 1,
        percentile: int = 95,
        prominence: float = 0.1,
        num_peaks: int = 3,
    ) -> np.ndarray | None:
        self.method = method
        if self.mode == "SPD" and self.spd_mode == 15:
            if method == "all_peaks":
                self.points = self._get_all_peak_points()
                return self.points
            elif method == "multi_peaks":
                self.points = self._get_multi_peak_points(num_peaks=num_peaks)
                return self.points
            elif method == "percent":
                # Convert top percentile% and all peak bins to point cloud
                self.points = self._get_percentile_peak_points()
                return self.points
            else:
                print(f"Calculating distances using method: {method}")
                self.distances = self._get_distance()
                self.intensity = self._calculate_and_normalize_intensity(
                    use_gain=True, gain=intensity_gain, stack_num=stack_num
                )
                self.points = self._calc_pc_coordinate(self.df)
                return self.points
        elif self.mode == "SPD" and self.spd_mode in [1, 5, 12]:
            self.distances = self._get_distance()
            self.points = self._calc_pc_coordinate(self.df)
            return self.points

    def _calculate_distance_from_bin_simple(self, bin_idx: int) -> float:
        """Calculate distance [m] from bin index (common logic)"""
        if self.mode == "SPD" and self.spd_mode == 15:
            return (bin_idx - self.histo_offset) * 10e-10 * 299792458 / 2
        return 0.0  # Fallback for other modes or if not SPD 15

    def get_distance_from_bin(self, hist: np.ndarray, peak_bin: int) -> DistanceResult:
        """
        Calculates distance using FWHM interpolation for a given histogram and peak.
        """
        return self._calculate_fwhm_distance(hist, peak_bin)

    def _get_histogram(self, df: pd.DataFrame) -> np.ndarray:
        def parse_hist_data(x: Union[str, np.ndarray]) -> np.ndarray:
            if isinstance(x, str):
                # Remove [] at the beginning and end, and remove extra spaces
                cleaned = x.strip().strip("[]")
                return np.array(list(map(int, cleaned.split())))
            elif isinstance(x, np.ndarray):
                return x
            else:
                raise ValueError(f"Unexpected data type in HistData column: {type(x)}")

        return np.array(df["HistData"].apply(parse_hist_data).tolist())

    def _calculate_fwhm_distance(self, hist: np.ndarray, peak_bin: int) -> DistanceResult:
        """
        Calculates distance based on Full Width at Half Maximum (FWHM) interpolation.
        """
        speed_of_light_per_bin = 10e-10 * 299792458 / 2  # m/bin
        uniform_offset = 1.20  # m

        peak_value = hist[peak_bin]
        half_max = peak_value / 2.0

        start_fwhm_bin: Optional[float] = None
        end_fwhm_bin: Optional[float] = None
        notes: List[str] = []

        # --- Find start_fwhm_bin (left side interpolation) ---
        # Search left from peak_bin
        left_bins = np.arange(peak_bin - 1, -1, -1)
        for i in left_bins:
            if hist[i] <= half_max:
                # Found crossing interval [i, i+1]
                # hist[i] <= half_max < hist[i+1]
                if hist[i + 1] - hist[i] == 0:  # Avoid division by zero
                    start_fwhm_bin = float(i)  # Fallback to integer bin
                    notes.append("start_fwhm_bin: flat region, no interpolation")
                else:
                    start_fwhm_bin = i + (half_max - hist[i]) / (hist[i + 1] - hist[i])
                break
        if start_fwhm_bin is None:
            # Half-max not found on left side (e.g., peak is at the beginning or histogram is rising)
            start_fwhm_bin = float(peak_bin)  # Fallback
            notes.append("start_fwhm_bin: half-max not found on left, using peak_bin")

        # --- Find end_fwhm_bin (right side interpolation) ---
        # Search right from peak_bin
        right_bins = np.arange(peak_bin + 1, len(hist))
        for i in right_bins:
            if hist[i] <= half_max:
                # Found crossing interval [i-1, i]
                # hist[i-1] > half_max >= hist[i]
                if hist[i] - hist[i - 1] == 0:  # Avoid division by zero
                    end_fwhm_bin = float(i)  # Fallback to integer bin
                    notes.append("end_fwhm_bin: flat region, no interpolation")
                else:
                    end_fwhm_bin = i - (half_max - hist[i]) / (hist[i - 1] - hist[i])
                break
        if end_fwhm_bin is None:
            # Half-max not found on right side
            end_fwhm_bin = float(peak_bin)  # Fallback
            notes.append("end_fwhm_bin: half-max not found on right, using peak_bin")

        # --- Calculate depth and distance ---
        depth: float
        fwhm_bins_list: Optional[List[float]] = None

        if start_fwhm_bin is not None and end_fwhm_bin is not None:
            depth = (start_fwhm_bin + end_fwhm_bin) / 2.0
            fwhm_bins_list = [start_fwhm_bin, end_fwhm_bin]
        else:
            # Fallback if FWHM calculation failed for some reason
            depth = float(peak_bin)
            notes.append("depth: FWHM calculation failed, using peak_bin")

        distance_m = depth * speed_of_light_per_bin - uniform_offset

        # Additional notes
        if peak_value >= np.iinfo(hist.dtype).max:  # Check for saturation
            notes.append("saturation: peak value reached max histogram count")
        if len(find_peaks(hist, height=half_max)[0]) > 1:  # Check for multiple peaks above half-max
            notes.append("multiple_peaks_above_half_max")
        if distance_m < 0:
            notes.append("negative_distance")

        return DistanceResult(
            distance_m=distance_m,
            start_fwhm_bin=start_fwhm_bin,
            peak_bin=peak_bin,
            fwhm_bins=fwhm_bins_list,
            notes="; ".join(notes),
        )

    def _process_histogram(
        self, hist, sigma, median_kernel_size, prominence_factor, distance, width
    ):
        """
        Processes a single histogram to find peaks using the multi-stage algorithm.
        Returns the processed data and detected peaks.
        """
        if hist.sum() == 0:
            return None, None, None, None

        # 1. Gaussian Smoothing
        smoothed_hist = gaussian_filter1d(hist.astype(np.float32), sigma=sigma)

        # 2. Median Filtering for baseline estimation
        baseline = medfilt(smoothed_hist, kernel_size=median_kernel_size)

        # 3. Baseline Subtraction
        residual = smoothed_hist - baseline
        residual[residual < 0] = 0

        # 4. Robust Standard Deviation (MAD) for prominence threshold
        median_residual = np.median(residual)
        mad = np.median(np.abs(residual - median_residual))

        if mad == 0:
            return smoothed_hist, baseline, residual, np.array([])

        # 5. Set thresholds
        prominence_threshold = mad * prominence_factor
        height_threshold = mad * 1.5

        # 6. Find peaks
        peaks, properties = find_peaks(
            residual,
            height=height_threshold,
            prominence=prominence_threshold,
            distance=distance,
            width=width,
        )

        return smoothed_hist, baseline, residual, peaks

    def _get_percentile_peak_points(
        self, sigma=1, median_kernel_size=51, prominence_factor=3, distance=5, width=1
    ) -> np.ndarray:
        """
        (Refactored)
        Use a multi-stage signal processing algorithm to detect peaks for each histogram,
        and convert all peaks to 3D points and return them.
        """
        all_points = []
        peak_values = []
        peak_bins = []

        histograms = self.histograms
        df = self.df

        for i, hist in enumerate(histograms):
            _, _, _, peaks = self._process_histogram(
                hist, sigma, median_kernel_size, prominence_factor, distance, width
            )

            if peaks is None or peaks.size == 0:
                continue

            row = df.iloc[i]
            x = int(row["//Pixel_X"])
            y = int(row["Pixel_Y"])

            for bin_idx in peaks:
                distance_result = self.get_distance_from_bin(hist, bin_idx)
                point = self._calc_single_point_spd(x, y, distance_result.distance_m)
                all_points.append(point)
                peak_values.append(hist[bin_idx])
                peak_bins.append(bin_idx)

        self.percent_points_peak_values = np.array(peak_values, dtype=float)
        self.percent_points_peak_bins = np.array(peak_bins, dtype=int)

        return np.array(all_points, dtype=float)

    def _get_all_peak_points(self) -> np.ndarray:
        """
        Use the median of each histogram as a threshold to detect peaks.
        - Maximum peak: OBJECT
        - Other peaks: NOISE
        Also, generate 3D point clouds for all peaks.
        """
        all_points = []
        intensities = []
        annotations = []

        histogram_medians = np.median(self.histograms, axis=1)

        for idx, hist in enumerate(self.histograms):
            row = self.df.iloc[idx]
            threshold = histogram_medians[idx]

            peaks, properties = find_peaks(hist, height=threshold)
            annotation = {}

            if len(peaks) > 0:
                peak_heights = properties["peak_heights"]
                max_peak = peaks[peak_heights.argmax()]
                for peak in peaks:
                    label = 1 if peak == max_peak else 0  # 1: OBJECT, 0: NOISE
                    annotation[int(peak)] = label

                    # Convert distance to point cloud coordinates for each peak position
                    distance_result = self.get_distance_from_bin(hist, peak)
                    point = self._calc_single_point(row, distance_result.distance_m)
                    all_points.append(point)
                    intensities.append(hist[peak])

            annotations.append(
                {"x": int(row["//Pixel_X"]), "y": int(row["Pixel_Y"]), "annotation": annotation}
            )

        self.intensity = intensities
        self.annotations = annotations  # For external reference (optional)
        return np.array(all_points)

    def _get_multi_peak_points(self, num_peaks: int) -> np.ndarray:
        """
        Use the median of each histogram as a threshold to detect peaks,
        and detect the specified number of peaks in descending order of intensity.
        """
        all_points = []
        intensities = []

        # Calculate median for each histogram to use as a threshold
        histogram_medians = np.median(self.histograms, axis=1)

        for idx, hist in enumerate(self.histograms):
            row = self.df.iloc[idx]

            # Set the threshold to the median of the current histogram
            threshold = histogram_medians[idx]

            # Find peaks above the threshold
            peaks, properties = find_peaks(hist, height=threshold)

            if len(peaks) > 0:
                peak_heights = properties["peak_heights"]

                # Sort peaks by height in descending order
                sorted_indices = np.argsort(peak_heights)[::-1]

                # Select the top 'num_peaks'
                top_peaks = peaks[sorted_indices[:num_peaks]]

                for peak in top_peaks:
                    # Convert each peak to a 3D point
                    distance_result = self.get_distance_from_bin(hist, peak)
                    point = self._calc_single_point(row, distance_result.distance_m)
                    all_points.append(point)
                    intensities.append(hist[peak])

        self.intensity = intensities
        return np.array(all_points)

    def _find_peak_time(self, hist_data: np.ndarray) -> np.ndarray:
        return np.array([np.argmax(data) for data in hist_data])

    def _find_centroid_time(self, hist_data: np.ndarray) -> np.ndarray:
        return np.array([np.sum(np.arange(len(data)) * data) / np.sum(data) for data in hist_data])

    def _find_average_time(self, hist_data: np.ndarray) -> np.ndarray:
        return np.array([np.mean(np.where(data > 0)[0]) for data in hist_data])

    def _find_first_peak_time(self, hist_data: np.ndarray) -> np.ndarray:
        """
        Select peaks from each histogram under the following conditions and return the time (index):
        1. Detect peaks with height greater than 3
        2. Select the top 3 peaks in descending order of height
        3. Select the peak that is "earliest in time" (smallest index) among the top 3
        4. If no peak is found, use the maximum value position

        Returns:
            np.ndarray: Peak position (time) for each point
        """
        peak_times = []
        for data in hist_data:
            peaks, properties = find_peaks(data, height=3)
            if len(peaks) == 0:
                # If no peak is found, use the maximum value position
                peak_times.append(int(np.argmax(data)))
                continue

            # Sort peaks by height in descending order
            peak_heights = properties["peak_heights"]
            sorted_indices = np.argsort(-peak_heights)  # Descending order
            top_peaks = peaks[sorted_indices[:2]]  # Select top 2

            # Select the peak that is "earliest in time" (smallest index) among the top 3
            first_peak = int(np.min(top_peaks))
            peak_times.append(first_peak)
        return np.array(peak_times)

    def _get_distance(self) -> np.ndarray:
        if self.spd_mode == 15:
            peak_bins: np.ndarray
            if self.method == "strongest" or self.method == "all_peaks" or self.method == "percent":
                peak_bins = self._find_peak_time(self.histograms)
            elif self.method == "centroid":
                peak_bins = self._find_centroid_time(self.histograms).astype(int)
            elif self.method == "average":
                peak_bins = self._find_average_time(self.histograms).astype(int)
            elif self.method == "first_peak":
                peak_bins = self._find_first_peak_time(self.histograms)
            else:
                raise ValueError(f"Unknown distance calculation method: {self.method}")

            # Now, for each histogram and its peak_bin, calculate DistanceResult
            distance_results: List[DistanceResult] = []
            for i, hist in enumerate(self.histograms):
                result = self.get_distance_from_bin(hist, peak_bins[i])
                distance_results.append(result)

            return np.array(
                distance_results, dtype=object
            )  # Return array of DistanceResult objects
        else:
            return self.df["Depth"].values

    def _calc_single_point(self, df_row: pd.Series, distance: float) -> list[float]:
        if self.mode == "SRL":
            horizontal_index = df_row["//Pixel_X"]
            vertical_index = df_row["Pixel_Y"]
            azimuth = (horizontal_index * 47 - 4512) / 100.0
            altitude = (vertical_index * 47 - 1316) / 100.0
            alpha = np.deg2rad(azimuth)
            omega = np.deg2rad(altitude)
            x = distance * np.cos(alpha) * np.cos(omega)
            y = -distance * np.sin(alpha) * np.cos(omega)
            z = -distance * np.sin(omega)
            return [x, y, z]

        elif self.mode == "SPD":
            if self.spd_mode in [1, 5, 12]:
                hfov = 120.0
                vfov = 25.6
            elif self.spd_mode == 15:
                hfov = 40.0
                vfov = 25.6
            else:
                raise ValueError("Unsupported SPD mode.")

            h_res = 0.1
            v_res = 0.05
            horizontal_index = df_row["//Pixel_X"]
            vertical_index = df_row["Pixel_Y"]
            azimuth = horizontal_index * h_res
            altitude = vfov - vertical_index * v_res
            alpha = np.deg2rad(azimuth)
            omega = np.deg2rad(altitude)
            x = distance * np.cos(alpha) * np.cos(omega)
            y = distance * np.sin(alpha) * np.cos(omega)
            z = -distance * np.sin(omega)
            return [x, y, z]

        else:
            raise ValueError(f"Unknown mode: {self.mode}")

    def _calc_pc_coordinate(self, df: pd.DataFrame) -> np.ndarray:
        points: list[list[float]] = []

        if self.mode == "SRL":
            for idx, distance_result in enumerate(self.distances):
                distance = distance_result.distance_m
                horizontal_index: int = df.iloc[idx]["//Pixel_X"]
                vertical_index: int = df.iloc[idx]["Pixel_Y"]
                azimuth: float = (horizontal_index * 47 - 4512) / 100.0
                altitude: float = (vertical_index * 47 - 1316) / 100.0
                alpha: float = np.deg2rad(azimuth)
                omega: float = np.deg2rad(altitude)
                x: float = distance * np.cos(alpha) * np.cos(omega)
                y: float = -distance * np.sin(alpha) * np.cos(omega)
                z: float = -distance * np.sin(omega)
                points.append([x, y, z])

        elif self.mode == "SPD":
            # SPDモードの座標変換
            for idx, distance_result in enumerate(self.distances):
                distance = distance_result.distance_m
                horizontal_index: int = df.iloc[idx]["//Pixel_X"]
                vertical_index: int = df.iloc[idx]["Pixel_Y"]
                point = self._calc_single_point_spd(horizontal_index, vertical_index, distance)
                points.append(point)
        #     # SPD Parameters
        #     if self.spd_mode in [1, 5, 12]:
        #         hfov = 120.0  # degrees
        #         vfov = 25.6
        #     elif self.spd_mode == 15:
        #         hfov = 40.0
        #         vfov = 25.6
        #     else:
        #         raise ValueError("Unsupported SPD mode.")

        #     h_res = 0.1  # horizontal resolution [deg]
        #     v_res = 0.05  # vertical resolution [deg]

        #     for idx, distance in enumerate(self.distances):
        #         horizontal_index: int = df.iloc[idx]['//Pixel_X']
        #         vertical_index: int = df.iloc[idx]['Pixel_Y']
        #         h_pixels = int(hfov / h_res)
        #         v_pixels = int(vfov / v_res)
        #         azimuth = -hfov / 2 + horizontal_index * h_res
        #         altitude = vfov / 2 - vertical_index * v_res  # invert Y

        #         alpha = np.deg2rad(azimuth)
        #         omega = np.deg2rad(altitude)
        #         x = distance * np.cos(alpha) * np.cos(omega)
        #         y = - distance * np.sin(alpha) * np.cos(omega)
        #         z = - distance * np.sin(omega)
        #         points.append([x, y, z])
        # else:
        #     raise ValueError(f"Unknown mode: {self.mode}")

        return np.array(points)

    # Left-handed coordinate system with Z direction as the camera direction
    def _calc_single_point_spd(self, x: int, y: int, distance: float) -> list:
        """
        Coordinate transformation from (x, y, distance) to (X, Y, Z) in SPD mode
        """
        h_res = 0.1
        v_res = 0.05

        azimuth = x * h_res
        altitude = y * v_res

        alpha = np.deg2rad(azimuth)
        omega = np.deg2rad(altitude)

        X = distance * np.cos(alpha) * np.cos(omega)
        Y = -distance * np.sin(alpha) * np.cos(omega)
        Z = -distance * np.sin(omega)
        return [X, Y, Z]

    def _get_spd_block_gain_map(self) -> np.ndarray:
        """
        Generate a gain map for SPD mode. 520 lines are divided into 8 blocks using Pixel_Y + 4.
        """
        pixel_y = self.df["Pixel_Y"].values
        block_gains = np.array(
            [8 / 1, 8 / 1, 8 / 2, 8 / 5, 8 / 8, 8 / 2, 8 / 1, 8 / 1]
        )  # Correction by inverse of magnification
        block_size = 64

        gain_map = np.ones_like(pixel_y, dtype=float)
        shifted_y = pixel_y + 4
        in_range = (shifted_y >= 0) & (shifted_y < block_size * len(block_gains))
        valid_shifted_y = shifted_y[in_range]
        block_indices = (valid_shifted_y // block_size).astype(int)
        block_indices = np.clip(block_indices, 0, len(block_gains) - 1)
        gain_map[in_range] = block_gains[block_indices]
        return gain_map

    def _calculate_and_normalize_intensity(
        self, gain, stack_num, use_gain: bool = True
    ) -> np.ndarray:
        """
        Detect peaks from the histogram, calculate intensity, and normalize it.
        - If there are consecutive peaks, add bonus (fixed value × number of consecutive peaks)
        - If SPD mode (1, 12, 15) and use_gain=True, multiply the gain by the block and then normalize
        """
        if self.method == "strongest":
            peak_indices = self._find_peak_time(self.histograms)
        elif self.method == "centroid":
            peak_indices = self._find_centroid_time(self.histograms).astype(int)
        elif self.method == "average":
            peak_indices = self._find_average_time(self.histograms).astype(int)
        elif self.method == "first_peak":
            peak_indices = self._find_first_peak_time(self.histograms)
        else:
            raise ValueError(f"Unknown method: {self.method}")

        # Save peak indices
        self.peak_indices = peak_indices

        # Get gain map
        if self.mode == "SPD" and self.spd_mode in [1, 12, 15] and use_gain:
            gain_map = self._get_spd_block_gain_map()
        else:
            gain_map = np.ones(len(self.histograms), dtype=float)

        adjusted = []
        for idx, (hist, peak_idx) in enumerate(zip(self.histograms, peak_indices)):
            if peak_idx >= len(hist):
                intensity = 0
            else:
                value = hist[peak_idx]
                left = peak_idx
                while left > 0 and hist[left - 1] == value:
                    left -= 1
                right = peak_idx
                while right < len(hist) - 1 and hist[right + 1] == value:
                    right += 1
                continuity_len = right - left
                bonus = continuity_len * gain
                intensity = value * gain_map[idx] + bonus
            adjusted.append(intensity)

        adjusted = np.array(adjusted) / stack_num
        # If the maximum value exceeds 255, limit it to 255
        if adjusted.max() > 255:
            adjusted = np.where(adjusted > 255, 255, adjusted)
        self.raw_intensity = adjusted.astype(int)
        # Maximum value for normalization (using the maximum value before clipping is also OK, here we use the maximum value after clipping)
        max_val = adjusted.max() or 1.0
        return adjusted / max_val

    def create_intensity_image(self, mode: str = "normalized") -> np.ndarray:
        if mode == "raw":
            intensity_data = self.raw_intensity
            dtype = int
        elif mode == "normalized":
            intensity_data = self.intensity
            dtype = float

        width = int(self.df["//Pixel_X"].max()) + 1
        height = int(self.df["Pixel_Y"].max()) + 1
        image = np.zeros((height, width), dtype=dtype)

        pixel_x = self.df["//Pixel_X"].values.astype(int)
        pixel_y = self.df["Pixel_Y"].values.astype(int)

        for x, y, val in zip(pixel_x, pixel_y, intensity_data):
            image[y, x] = val

        print(f"max intensity value in image: {image.max()}")
        print(f"min intensity value in image: {image.min()}")

        return image

    def create_depth_image(self, mode: str = "raw") -> np.ndarray:
        """
        Calculate distance from the histogram and generate a depth image.
        - mode: "raw" or "normalized"
        Returns:
            np.ndarray: Depth image (H, W)
        """
        if mode == "raw":
            depth_data = self.distances
            dtype = float
        elif mode == "normalized":
            depth_data = self._get_distance()
            dtype = float
        else:
            raise ValueError(f"Unknown mode: {mode}")

        width = int(self.df["//Pixel_X"].max()) + 1
        height = int(self.df["Pixel_Y"].max()) + 1
        image = np.zeros((height, width), dtype=dtype)

        pixel_x = self.df["//Pixel_X"].values.astype(int)
        pixel_y = self.df["Pixel_Y"].values.astype(int)

        for x, y, val in zip(pixel_x, pixel_y, depth_data):
            image[y, x] = val

        print(f"max depth value in image: {image.max()}")
        print(f"min depth value in image: {image.min()}")

        return image

    def normalize_histograms(self, use_gain: bool = True, stack_num: int = 1) -> np.ndarray:
        """
        Apply the same gain map as intensity to all bins of each histogram,
        divide by stack_num, and then normalize by the maximum value and return.

        Returns:
            np.ndarray: shape = (N_pixels, N_bins), normalized histograms
        """
        histograms = self.histograms.copy().astype(float)

        # Get gain map
        if self.mode == "SPD" and self.spd_mode in [1, 12, 15] and use_gain:
            gain_map = self._get_spd_block_gain_map()
        else:
            gain_map = np.ones(len(histograms), dtype=float)

        # Apply gain to each histogram
        for idx in range(len(histograms)):
            histograms[idx] *= gain_map[idx]

        # Normalize by the maximum value (row-wise)
        max_vals = 18 * 8 * stack_num
        normalized = histograms / max_vals

        return normalized

    def visualize_image_with_histograms(
        self,
        image: np.ndarray,
        sigma=1,
        median_kernel_size=51,
        prominence_factor=3,
        distance=5,
        width=1,
    ):
        fig, ax = plt.subplots(figsize=(10, 5))
        im = ax.imshow(image, cmap="hot")
        plt.colorbar(im, ax=ax, label="Intensity")
        plt.title("Intensity Image (Click a point to view its histogram)")
        plt.xlabel("Pixel X")
        plt.ylabel("Pixel Y")

        # Use raw histograms
        histgrams = self.histograms

        def onclick(event):
            if event.inaxes != ax:
                return
            x = int(event.xdata)
            y = int(event.ydata)
            print(f"Clicked on: X={x}, Y={y}")
            self.plot_histogram_for_pixel(
                histgrams, x, y, sigma, median_kernel_size, prominence_factor, distance, width
            )

        fig.canvas.mpl_connect("button_press_event", onclick)
        plt.show()

    def plot_histogram_for_pixel(
        self,
        histgrams,
        x: int,
        y: int,
        sigma=1,
        median_kernel_size=51,
        prominence_factor=3,
        distance=5,
        width=1,
    ):
        """
        Plot the histogram and the result of the peak detection algorithm for the specified pixel coordinates (x, y).
        """
        match = self.df[(self.df["//Pixel_X"] == x) & (self.df["Pixel_Y"] == y)]
        if match.empty:
            print(f"No data found for the specified coordinates x={x}, y={y}.")
            return

        # The `histgrams` passed here should be the raw histograms
        hist_data = histgrams[match.index[0]]

        smoothed_hist, baseline, residual, peaks = self._process_histogram(
            hist_data, sigma, median_kernel_size, prominence_factor, distance, width
        )

        if smoothed_hist is None:
            print(f"No data to process for pixel ({x}, {y}).")
            return

        plt.close("hist_figure")
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True, num="hist_figure")
        fig.suptitle(f"Histogram Processing at Pixel ({x}, {y})")

        # Top plot: Raw, Smoothed, and Baseline
        ax1.plot(hist_data, label="Raw Histogram", color="gray", alpha=0.7)
        ax1.plot(smoothed_hist, label="Smoothed", color="blue")
        ax1.plot(baseline, label="Estimated Baseline", color="green", linestyle="--")
        ax1.set_title("1. Smoothing and Baseline Estimation")
        ax1.set_ylabel("Count")
        ax1.legend()
        ax1.grid(True)

        # Bottom plot: Residual and Detected Peaks
        ax2.plot(residual, label="Residual (Smoothed - Baseline)", color="purple")
        if peaks.size > 0:
            ax2.plot(peaks, residual[peaks], "x", label="Detected Peaks", color="red", markersize=8)
        ax2.set_title("2. Peak Detection on Residual")
        ax2.set_xlabel("Bin Index")
        ax2.set_ylabel("Count")
        ax2.legend()
        ax2.grid(True)

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()

    def visualize_intensity_image(self, image: np.ndarray, threshold_percentile=93):
        """
        Display the image and plot the histogram of the clicked pixel with threshold lines and vertical lines for bins above the threshold.
        """
        fig, ax = plt.subplots(figsize=(10, 5))
        im = ax.imshow(image, cmap="hot")
        plt.colorbar(im, ax=ax, label="Normalized Intensity")
        plt.title("Intensity Image (Click a point to view its histogram)")
        plt.xlabel("Pixel X")
        plt.ylabel("Pixel Y")

        # Normalized histograms to use when clicking (OK to use original histograms if needed)
        histgrams = self.normalize_histograms(use_gain=True, stack_num=10)

        def onclick(event):
            if event.inaxes != ax:
                return
            x = int(event.xdata)
            y = int(event.ydata)
            print(f"Clicked on: X={x}, Y={y}")
            self.plot_histogram_for_pixel(histgrams, x, y, threshold_percentile)

        fig.canvas.mpl_connect("button_press_event", onclick)
        plt.tight_layout()
        plt.show()
