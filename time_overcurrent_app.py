import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import MultipleLocator
import streamlit.components.v1 as components # NEW IMPORT

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
    # MODIFIED: Removed layout="wide" to use Streamlit's default narrow/centered layout for better mobile fit
)

st.header("Time-Overcurrent Relay (IDMT) Simulator")
st.markdown("Use the sidebar to adjust relay settings and visualize the tripping curve.")

# --- 3.1. TTS Component (NEW SECTION) ---
NARRATIVE_TEXT = """
Welcome to the Time Overcurrent Relay Simulator. This tool visualizes the inverse definite minimum time, or IDMT, characteristic curve, which is the core of modern power system protection. The curve shows the trip time needed for a given fault current. 

On the vertical axis, you have the operating time in milliseconds, and on the horizontal axis, the fault current in amperes. You can control the two critical settings using the sliders in the sidebar. 

First, the Pickup Current, or I-pickup, defines the minimum current required to start the relay timer. If the fault current is below this value, the relay will not trip. 

Second, the Time Multiplier Setting, or TMS, shifts the curve vertically. A higher TMS means the relay will take longer to trip for any given fault current. 

By changing these sliders, you adjust the coordination time, which is essential for safely isolating faults in the grid.
"""

TTS_HTML_COMPONENT = f"""
<audio id="audioPlayer" controls style="width: 100%; display:none; margin-top: 10px;"></audio>
<button id="narrateButton" 
        style="padding: 8px 16px; border-radius: 4px; background-color: #10b981; color: white; border: none; cursor: pointer; font-weight: bold;">
    ▶️ Start Narration
</button>
<p id="status" style="margin-top: 5px; font-size: 0.85em; color: #888;">Click to generate narration.</p>

<script>
    // Constants and Setup
    const NARRATIVE_TEXT = `{NARRATIVE_TEXT}`;
    // The API URL is appended with the key during runtime by the environment.
    const API_URL = 'https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-tts:generateContent?key='; 

    const button = document.getElementById('narrateButton');
    const audioPlayer = document.getElementById('audioPlayer');
    const status = document.getElementById('status');
    const VOICE_NAME = 'Rasalgethi'; // Knowledgeable voice

    function base64ToArrayBuffer(base64) {{
        const binaryString = atob(base64);
        const len = binaryString.length;
        const bytes = new Uint8Array(len);
        for (let i = 0; i < len; i++) {{
            bytes[i] = binaryString.charCodeAt(i);
        }}
        return bytes.buffer;
    }}

    function pcmToWav(pcm16, sampleRate = 16000) {{
        // API returns signed PCM16 data
        const numChannels = 1;
        const bitsPerSample = 16;
        const byteRate = sampleRate * numChannels * (bitsPerSample / 8);
        const blockAlign = numChannels * (bitsPerSample / 8);
        const dataSize = pcm16.byteLength;
        const buffer = new ArrayBuffer(44 + dataSize);
        const view = new DataView(buffer);
        let offset = 0;

        // Helper to write string to DataView
        function writeString(s) {{
            for (let i = 0; i < s.length; i++) {{
                view.setUint8(offset + i, s.charCodeAt(i));
            }}
            offset += s.length;
        }}

        // RIFF chunk
        writeString('RIFF');
        view.setUint32(offset, 36 + dataSize, true); offset += 4;
        writeString('WAVE');

        // fmt sub-chunk
        writeString('fmt ');
        view.setUint32(offset, 16, true); offset += 4; // Subchunk1Size
        view.setUint16(offset, 1, true); offset += 2; // AudioFormat (1 for PCM)
        view.setUint16(offset, numChannels, true); offset += 2; // NumChannels
        view.setUint32(offset, sampleRate, true); offset += 4; // SampleRate
        view.setUint32(offset, byteRate, true); offset += 4; // ByteRate
        view.setUint16(offset, blockAlign, true); offset += 2; // BlockAlign
        view.setUint16(offset, bitsPerSample, true); offset += 2; // BitsPerSample

        // data sub-chunk
        writeString('data');
        view.setUint32(offset, dataSize, true); offset += 4; // Subchunk2Size

        // Write PCM data
        for (let i = 0; i < pcm16.length; i++, offset += 2) {{
            view.setInt16(offset, pcm16[i], true);
        }}

        return new Blob([view], {{ type: 'audio/wav' }});
    }}

    async function generateNarrativeAudio() {{
        button.disabled = true;
        button.textContent = "Generating audio (1/3)...";
        status.textContent = "Calling Gemini TTS API...";
        audioPlayer.style.display = 'none';
        
        const payload = {{
            contents: [{{ parts: [{{ text: NARRATIVE_TEXT }}] }}],
            generationConfig: {{
                responseModalities: ["AUDIO"],
                speechConfig: {{
                    voiceConfig: {{
                        prebuiltVoiceConfig: {{ voiceName: VOICE_NAME }}
                    }}
                }}
            }},
            model: "gemini-2.5-flash-preview-tts"
        }};

        let attempts = 0;
        const maxRetries = 3;
        const baseDelay = 1000;

        while (attempts < maxRetries) {{
            try {{
                const response = await fetch(API_URL, {{
                    method: 'POST',
                    headers: {{ 'Content-Type': 'application/json' }},
                    body: JSON.stringify(payload)
                }});

                if (!response.ok) {{
                    if (response.status === 429) {{ // Rate Limit
                        throw new Error("Rate limit exceeded.");
                    }}
                    throw new Error(`API call failed: ${{response.status}} ${{response.statusText}}`);
                }}
                
                const result = await response.json();
                
                const part = result?.candidates?.[0]?.content?.parts?.[0];
                const audioData = part?.inlineData?.data;
                const mimeType = part?.inlineData?.mimeType;

                if (audioData && mimeType && mimeType.startsWith("audio/L16")) {{
                    status.textContent = "Processing audio data (2/3)...";
                    const pcmDataBuffer = base64ToArrayBuffer(audioData);
                    const pcm16 = new Int16Array(pcmDataBuffer);
                    const sampleRate = 16000; 

                    status.textContent = "Converting to WAV (3/3)...";
                    const wavBlob = pcmToWav(pcm16, sampleRate);
                    const audioUrl = URL.createObjectURL(wavBlob);
                    
                    audioPlayer.src = audioUrl;
                    audioPlayer.style.display = 'block';
                    button.textContent = "🔊 Narration Ready";
                    status.textContent = "Press play above to listen to the explanation.";
                    return; // Success, exit function
                }} else {{
                    throw new Error("Invalid or missing audio data in API response.");
                }}

            }} catch (error) {{
                attempts++;
                if (attempts >= maxRetries) {{
                    status.textContent = `Final Error: ${{error.message}}. Generation failed.`;
                }} else {{
                    const delay = baseDelay * (2 ** (attempts - 1));
                    // Note: In this environment, we avoid console logging retries.
                    status.textContent = `Attempt ${attempts}/${maxRetries} failed. Retrying in ${delay / 1000}s...`;
                    await new Promise(resolve => setTimeout(resolve, delay));
                }}
            }}
        }}

    }}

    button.addEventListener('click', generateNarrativeAudio);

</script>
"""
components.html(TTS_HTML_COMPONENT, height=120)

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
        max_value=2000.0,
        value=400.0, 
        step=5.0,
        key="fault_current_input"
    )

# --- 5. Main Calculation and Results (Condensed) ---

# Calculate PSM
psm_test = i_fault / i_pickup
trip_time_sec = calculate_trip_time(psm_test, tms, curve)

st.subheader("Results for Test Fault")
# Streamlit will naturally stack columns vertically on small screens
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

st.subheader(f"Characteristic Curve: {curve['name']}")

# Define plot ranges in Amperes based on I_pickup
max_psm_limit = 10.0 # Common max limit for plotting IDMT curves
max_time_plot = 10.0 
min_current_plot_start = i_pickup 
max_current_plot_limit = 2000.0 
max_current_plot = min(i_pickup * max_psm_limit, max_current_plot_limit)

# Generate I_fault points for the curve plot
current_values = np.linspace(min_current_plot_start, max_current_plot, 100)
psm_for_plot = current_values / i_pickup
curve_times = [calculate_trip_time(psm, tms, curve) for psm in psm_for_plot]

if len(curve_times) > 0:
    curve_times[0] = max_time_plot

curve_times_clamped = np.clip(curve_times, a_min=0, a_max=max_time_plot)


# Create the plot
fig, ax = plt.subplots(figsize=(5, 2.5)) # MODIFIED: Reduced figure width to 5 inches for mobile view

# Plot the IDMT Curve (X-axis is now current_values in Amperes)
ax.plot(current_values, curve_times_clamped * 1000, 
        label=curve["name"], 
        color="#10b981", 
        linewidth=3)

# Plot the specific Test Fault Point
if psm_test > 1.0:
    trip_time_to_plot_sec = min(trip_time_sec, max_time_plot)
    trip_time_to_plot_ms = trip_time_to_plot_sec * 1000
    plot_current = min(i_fault, max_current_plot)
    
    ax.plot(plot_current, trip_time_to_plot_ms, 'o', 
            color="#3b82f6", 
            markersize=8, 
            label="Test Fault Point")
    
    # Add annotation for the Current/Time
    if plot_current <= max_current_plot and trip_time_to_plot_ms < max_time_plot * 1000:
        ax.annotate(
            f'{i_fault:.0f} A',
            (plot_current, trip_time_to_plot_ms),
            textcoords="offset points",
            xytext=(10, 5),
            ha='center',
            fontsize=8
        )

# Add a vertical line for the Pickup Current (I_pickup)
ax.axvline(x=i_pickup, color='r', linestyle='--', label=f'$I_{{pickup}}$ ({i_pickup:.0f} A)')

# Style the plot
ax.set_title("Operating Time vs. Fault Current (I_fault)", fontsize=10)
ax.set_xlabel("Fault Current ($I_{fault}$, A)", fontsize=10)
ax.set_ylabel("Operating Time (ms)", fontsize=10)

# Set axis limits
x_lim_max = max(max_current_plot, min_current_plot_start + 100)
ax.set_xlim(i_pickup, x_lim_max)
ax.set_ylim(0, max_time_plot * 1000)

# Update X and Y-axis ticks
ax.xaxis.set_major_locator(MultipleLocator(500))
ax.xaxis.set_minor_locator(MultipleLocator(100))

ax.yaxis.set_major_locator(MultipleLocator(1000))
ax.yaxis.set_minor_locator(MultipleLocator(200))

ax.grid(True, linestyle='--', alpha=0.6)
ax.legend(fontsize=8)

# Display the plot in Streamlit
st.pyplot(fig)
