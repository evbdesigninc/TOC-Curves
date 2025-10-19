import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# --- 1. Relay Constants (Translated from JavaScript) ---
# Formula: t = TMS * [ A / ((PSM)^P - 1) ]
CURVE_CONSTANTS = {
    "normal_inverse": {"A": 0.14, "P": 0.02, "name": "IEC/IEEE Normal Inverse"},
    "very_inverse": {"A": 13.5, "P": 1.0, "name": "IEC/IEEE Very Inverse"},
    "extremely_inverse": {"A": 80.0, "P": 2.0, "name": "IEC/IEEE Extremely Inverse"},
    "long_time_inverse": {"A": 120.0, "P": 1.0, "name": "IEC Long Time Inverse"},
}

# --- 2. Calculation Logic (Translated from JavaScript) ---
def calculate_trip_time(psm, tms, constants):
    """
    Calculates the operating time (t) for a given PSM, TMS, and curve constants.
    Returns time in seconds, or np.inf if PSM <= 1.0.
    """
    if psm <= 1.0:
        return np.inf
    
    # Universal IDMT formula: t = TMS * [ A / ((PSM)^P - 1) ]
    denominator = psm**constants["P"] - 1
    
    # Check for near-zero denominator to prevent division error when PSM is ~1.0
    if denominator < 1e-6:
        return np.inf
        
    trip_time = tms * (constants["A"] / denominator)
    return trip_time

# --- 3. Streamlit Application Layout ---

st.set_page_config(
    page_title="Time Overcurrent Relay Simulator",
    layout="wide"
)

st.title("Time-Overcurrent Relay (IDMT) Simulator")
st.markdown("Use the sidebar to adjust relay settings and visualize the tripping curve.")

# --- 4. Input Sidebar ---
with st.sidebar:
    st.header("Relay Settings")
    
    # Curve Selection
    curve_key = st.selectbox(
        "Characteristic Curve Type:",
        options=list(CURVE_CONSTANTS.keys()),
        format_func=lambda x: CURVE_CONSTANTS[x]["name"],
        key="curve_select"
    )
    curve = CURVE_CONSTANTS[curve_key]

    st.subheader("Current Settings")
    
    # Pickup Current (I_pickup)
    i_pickup = st.slider(
        "Pickup Current ($I_{pickup}$, A):",
        min_value=1.0, 
        max_value=200.0, 
        value=50.0, 
        step=1.0,
        key="pickup_current_input"
    )

    # Time Multiplier Setting (TMS)
    # Using the adjusted range for better visualization of fast trips
    tms = st.slider(
        "Time Multiplier Setting (TMS):",
        min_value=0.05, 
        max_value=0.30, 
        value=0.15, 
        step=0.01,
        key="tms_slider",
        format="%.2f"
    )

    # Fault Current (I_fault)
    i_fault = st.slider(
        "Fault Current ($I_{fault}$, A):",
        min_value=1.0, 
        max_value=5000.0, 
        value=400.0, 
        step=5.0,
        key="fault_current_input"
    )

# --- 5. Main Calculation and Results ---

# Calculate PSM
psm_test = i_fault / i_pickup
trip_time_sec = calculate_trip_time(psm_test, tms, curve)

# Display Results (using Streamlit metrics)
st.subheader("Results for Test Fault")
col1, col2, col3 = st.columns(3)

col1.metric("Pickup Current", f"{i_pickup:.0f} A")
col2.metric("Fault Current", f"{i_fault:.0f} A")
col3.metric("PSM ($I_{fault} / I_{pickup}$)", f"{psm_test:.2f}")

st.divider()

# Display Trip Time Status
if psm_test <= 1.0:
    st.error("NO TRIP: Fault Current is below Pickup Current.")
else:
    trip_time_ms = trip_time_sec * 1000
    if np.isinf(trip_time_sec):
        st.warning("Trip Time is theoretically infinite (PSM is too close to 1.0).")
    else:
        st.success(f"Trip Time: **{trip_time_ms:,.0f} ms** (or {trip_time_sec:.3f} s)")

st.divider()

# --- 6. Plotting the Curve ---

st.header(f"Characteristic Curve: {curve['name']}")

# Define plot ranges
min_psm_plot = 1.1
max_psm_plot = 10.0
# Max time for the y-axis (1500 ms = 1.5 s, matching the HTML/JS version)
max_time_plot = 1.5 

# Generate PSM points for the curve plot
psm_values = np.linspace(min_psm_plot, max_psm_plot, 100)
curve_times = [calculate_trip_time(psm, tms, curve) for psm in psm_values]

# Clamp curve times to the max plot time
curve_times_clamped = np.clip(curve_times, a_min=0, a_max=max_time_plot)


# Create the plot
fig, ax = plt.subplots(figsize=(10, 5))

# Plot the IDMT Curve
ax.plot(psm_values, curve_times_clamped * 1000, 
        label=curve["name"], 
        color="#10b981", 
        linewidth=3)

# Plot the specific Test Fault Point
if psm_test > 1.0:
    plot_time_ms = min(trip_time_sec, max_time_plot) * 1000
    plot_psm = min(psm_test, max_psm_plot)
    
    ax.plot(plot_psm, plot_time_ms, 'o', 
            color="#3b82f6", 
            markersize=8, 
            label="Test Fault Point")
    
    # Add annotation for the PSM/Time if it's within plot bounds
    if plot_psm < max_psm_plot and plot_time_ms < max_time_plot * 1000:
        ax.annotate(
            f'PSM: {psm_test:.2f}',
            (plot_psm, plot_time_ms),
            textcoords="offset points",
            xytext=(10, 5),
            ha='center'
        )

# Style the plot
ax.set_title("Operating Time vs. Plug Setting Multiplier (PSM)")
ax.set_xlabel("PSM (I_fault / I_pickup)", fontsize=12)
ax.set_ylabel("Operating Time (ms)", fontsize=12)

# Set axis limits
ax.set_xlim(min_psm_plot, max_psm_plot)
ax.set_ylim(0, max_time_plot * 1000)

ax.grid(True, linestyle='--', alpha=0.6)
ax.legend()

# Display the plot in Streamlit
st.pyplot(fig)
