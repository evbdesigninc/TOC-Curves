import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import MultipleLocator

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
        max_value=2000.0,  # ADJUSTED: Max fault current set to 2000.0 A
        value=400.0, 
        step=5.0,
        key="fault_current_input"
    )

# --- 5. Main Calculation and Results (Condensed) ---

# Calculate PSM
psm_test = i_fault / i_pickup
trip_time_sec = calculate_trip_time(psm_test, tms, curve)

st.subheader("Results for Test Fault")
col1, col2, col3 = st.columns(3)

col1.metric("Pickup Current", f"{i_pickup:.0f} A")
col2.metric("Fault Current", f"{i_fault:.0f} A")
col3.metric("PSM ($I_{fault} / I_{pickup}$)", f"{psm_test:.2f}")

# Display Trip Time Status (Consolidated Block)
if psm_test <= 1.0:
    st.error("NO TRIP: Fault Current is below Pickup Current.")
else:
    trip_time_ms = trip_time_sec * 1000
    if np.isinf(trip_time_sec):
        st.warning("Trip Time is theoretically infinite (PSM is too close to 1.0).")
    else:
        # Using st.info for a visually clean, compact result
        st.info(f"Trip Time: **{trip_time_ms:,.0f} ms** (or {trip_time_sec:.3f} s)")

# --- 6. Plotting the Curve (Updated to use Current on X-axis) ---

st.header(f"Characteristic Curve: {curve['name']}")

# Define plot ranges in Amperes based on I_pickup
max_psm_limit = 10.0 # Common max limit for plotting IDMT curves

# ADJUSTMENT 1: Max time for the y-axis (10.0 s or 10000 ms)
max_time_plot = 10.0 

# Define the starting current for the curve data exactly at I_pickup
min_current_plot_start = i_pickup 

# ADJUSTMENT 2: Hard limit of 2000 A for the X-axis maximum
max_current_plot_calc = i_pickup * max_psm_limit 
max_current_plot_limit = 2000.0 
max_current_plot = min(max_current_plot_calc, max_current_plot_limit)

# Generate I_fault points for the curve plot
current_values = np.linspace(min_current_plot_start, max_current_plot, 100)

# Calculate PSM for each current point
psm_for_plot = current_values / i_pickup

# Calculate times based on the new PSM values
curve_times = [calculate_trip_time(psm, tms, curve) for psm in psm_for_plot]

# **CRITICAL CHANGE**: Force the time at the exact I_pickup point (index 0) to max time
# This ensures the characteristic curve visually starts at the top of the plot line.
if len(curve_times) > 0:
    curve_times[0] = max_time_plot

# Clamp all curve times to the max plot time
curve_times_clamped = np.clip(curve_times, a_min=0, a_max=max_time_plot)


# Create the plot
fig, ax = plt.subplots(figsize=(9, 3.5))

# Plot the IDMT Curve (X-axis is now current_values in Amperes)
ax.plot(current_values, curve_times_clamped * 1000, 
        label=curve["name"], 
        color="#10b981", 
        linewidth=3)

# Plot the specific Test Fault Point
if psm_test > 1.0:
    # Use the same clamping logic for the single point plot
    trip_time_to_plot_sec = min(trip_time_sec, max_time_plot)
    trip_time_to_plot_ms = trip_time_to_plot_sec * 1000
    
    # Clamp I_fault for plotting if it exceeds the max current plot range
    plot_current = min(i_fault, max_current_plot)
    
    ax.plot(plot_current, trip_time_to_plot_ms, 'o', 
            color="#3b82f6", 
            markersize=8, 
            label="Test Fault Point")
    
    # Add annotation for the Current/Time
    # Only annotate if the point is within the visible plot boundaries
    if plot_current <= max_current_plot and trip_time_to_plot_ms < max_time_plot * 1000:
        ax.annotate(
            f'{i_fault:.0f} A',
            (plot_current, trip_time_to_plot_ms),
            textcoords="offset points",
            xytext=(10, 5),
            ha='center'
        )

# Add a vertical line for the Pickup Current (I_pickup)
ax.axvline(x=i_pickup, color='r', linestyle='--', label=f'$I_{{pickup}}$ ({i_pickup:.0f} A)')

# Style the plot
ax.set_title("Operating Time vs. Fault Current (I_fault)")
ax.set_xlabel("Fault Current ($I_{fault}$, A)", fontsize=12)
ax.set_ylabel("Operating Time (ms)", fontsize=12)

# Set axis limits
x_lim_max = max(max_current_plot, min_current_plot_start + 100) # Ensure a minimum visible plot width
ax.set_xlim(i_pickup, x_lim_max) # Start X-axis visibly at I_pickup
ax.set_ylim(0, max_time_plot * 1000)

# ADJUSTMENT 3: Update X and Y-axis ticks
ax.xaxis.set_major_locator(MultipleLocator(400)) # Major ticks every 400 A
ax.xaxis.set_minor_locator(MultipleLocator(100)) # Minor ticks every 100 A

ax.yaxis.set_major_locator(MultipleLocator(1000)) # Major ticks every 1000 ms (1 second)
ax.yaxis.set_minor_locator(MultipleLocator(200)) # Minor ticks every 200 ms

ax.grid(True, linestyle='--', alpha=0.6)
ax.legend()

# Display the plot in Streamlit
st.pyplot(fig)
