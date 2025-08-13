import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import re

# --- Hydrophobicity Scales ---
# Define various hydrophobicity scales as dictionaries.
# This makes it easy to add more scales in the future.

KYTE_DOOLITTLE = {
    'A': 1.8, 'R': -4.5, 'N': -3.5, 'D': -3.5, 'C': 2.5,
    'Q': -3.5, 'E': -3.5, 'G': -0.4, 'H': -3.2, 'I': 4.5,
    'L': 3.8, 'K': -3.9, 'M': 1.9, 'F': 2.8, 'P': -1.6,
    'S': -0.8, 'T': -0.7, 'W': -0.9, 'Y': -1.3, 'V': 4.2
}

HOPP_WOODS = {
    'A': -0.5, 'R': 3.0, 'N': 0.2, 'D': 3.0, 'C': -1.0,
    'Q': 0.2, 'E': 3.0, 'G': 0.0, 'H': -0.5, 'I': -1.8,
    'L': -1.8, 'K': 3.0, 'M': -1.3, 'F': -2.5, 'P': 0.0,
    'S': 0.3, 'T': -0.4, 'W': -3.4, 'Y': -2.3, 'V': -1.5
}

EISENBERG = {
    'A': 0.62, 'R': -2.53, 'N': -0.78, 'D': -0.90, 'C': 0.29,
    'Q': -0.85, 'E': -0.74, 'G': 0.48, 'H': -0.40, 'I': 1.38,
    'L': 1.06, 'K': -1.50, 'M': 0.64, 'F': 1.19, 'P': 0.12,
    'S': -0.18, 'T': -0.05, 'W': 0.81, 'Y': 0.26, 'V': 1.08
}

SCALES = {
    "Kyte & Doolittle": KYTE_DOOLITTLE,
    "Hopp & Woods": HOPP_WOODS,
    "Eisenberg": EISENBERG,
}

# --- Core Functions ---

def validate_protein_sequence(sequence):
    """
    Checks if the input sequence contains only valid amino acid characters.
    Returns a set of invalid characters found, or an empty set if valid.
    """
    valid_chars = set(KYTE_DOOLITTLE.keys())
    sequence_chars = set(sequence.upper())
    invalid_chars = sequence_chars - valid_chars
    return invalid_chars

def calculate_hydrophobicity(sequence, scale, window_size):
    """
    Calculates hydrophobicity scores along the sequence using a sliding window.

    Args:
        sequence (str): The protein sequence.
        scale (dict): The hydrophobicity scale to use.
        window_size (int): The size of the sliding window.

    Returns:
        tuple: A tuple containing two lists: window positions and their
               corresponding average hydrophobicity scores.
    """
    scores = [scale.get(aa, 0) for aa in sequence.upper()]
    window_scores = []
    window_positions = []

    # The effective window size must be odd to have a central residue
    if window_size % 2 == 0:
        window_size += 1

    half_window = window_size // 2

    for i in range(len(scores) - window_size + 1):
        window = scores[i:i + window_size]
        avg_score = np.mean(window)
        window_scores.append(avg_score)
        # Position is the central residue of the window
        window_positions.append(i + half_window + 1)

    return window_positions, window_scores

def create_plot(positions, scores, scale_name, window_size, sequence_length):
    """
    Generates a professional-looking Matplotlib plot of the hydrophobicity scores.
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    ax.plot(positions, scores, color='#007acc', linewidth=2, label=f'{scale_name} Hydrophobicity')

    # Add a horizontal line at y=0 to distinguish hydrophobic/hydrophilic
    ax.axhline(0, color='gray', linestyle='--', linewidth=1)

    # --- Styling ---
    ax.set_title(f'Hydrophobicity Plot ({scale_name})', fontsize=16, fontweight='bold')
    ax.set_xlabel('Amino Acid Position', fontsize=12)
    ax.set_ylabel('Average Hydrophobicity Score', fontsize=12)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax.set_xlim(1, sequence_length)

    # Add text for hydrophobic/hydrophilic regions
    plot_max_y = max(scores) if scores else 1
    plot_min_y = min(scores) if scores else -1
    ax.text(ax.get_xlim()[1] * 0.98, plot_max_y, 'Hydrophobic →',
            verticalalignment='top', horizontalalignment='right',
            color='red', fontsize=10)
    ax.text(ax.get_xlim()[1] * 0.98, plot_min_y, '← Hydrophilic',
            verticalalignment='bottom', horizontalalignment='right',
            color='blue', fontsize=10)


    # Set plot background and border
    fig.patch.set_facecolor('#f4f4f4')
    ax.set_facecolor('white')
    plt.tight_layout()

    return fig

# --- Streamlit UI ---

def main():
    """
    The main function that runs the Streamlit application.
    """
    st.set_page_config(page_title="Protein Hydrophobicity Plotter", layout="wide")

    # --- Header ---
    st.title("🧬 Protein Hydrophobicity Plot Generator")
    st.markdown("""
    This tool generates a hydrophobicity plot for a given protein sequence.
    Hydrophobicity plots are useful for predicting membrane-spanning regions,
    antigenic sites, and regions of a protein that are likely to be exposed on the surface.
    """)
    st.markdown("---")

    # --- Sidebar for Controls ---
    st.sidebar.header("⚙️ Plot Controls")
    selected_scale_name = st.sidebar.selectbox(
        "Choose a Hydrophobicity Scale:",
        list(SCALES.keys())
    )
    window_size = st.sidebar.slider(
        "Select Window Size:",
        min_value=3,
        max_value=25,
        value=9,
        step=2, # Keep it odd for a central residue
        help="The number of amino acids to average over for each point in the plot. Odd numbers are recommended."
    )

    # --- Main Panel for Input and Output ---
    st.header("1. Enter Protein Sequence")
    sequence_input = st.text_area(
        "Paste your protein sequence in FASTA format or as a raw sequence:",
        height=200,
        placeholder=">sp|P0DP23|VIME_HUMAN\nMSFSTSV...",
    )

    if st.button("Generate Plot", type="primary"):
        if not sequence_input:
            st.warning("Please enter a protein sequence.")
            st.stop()

        # --- Sequence Processing ---
        # Remove FASTA header if present
        if sequence_input.startswith('>'):
            sequence = "".join(sequence_input.split('\n')[1:])
        else:
            sequence = sequence_input

        # Clean up any whitespace or numbers
        sequence = re.sub(r'[\s\d]', '', sequence).upper()

        if not sequence:
            st.error("The input contained no valid sequence data after cleaning.")
            st.stop()

        # --- Validation ---
        invalid_chars = validate_protein_sequence(sequence)
        if invalid_chars:
            st.error(f"Invalid characters found in sequence: {', '.join(invalid_chars)}")
            st.info("Please provide a sequence with standard single-letter amino acid codes.")
            st.stop()

        # --- Calculation and Plotting ---
        with st.spinner('Calculating scores and generating plot...'):
            selected_scale = SCALES[selected_scale_name]
            positions, scores = calculate_hydrophobicity(sequence, selected_scale, window_size)

            if not positions:
                st.error(f"Sequence is too short for the selected window size of {window_size}.")
                st.stop()

            st.header("2. Hydrophobicity Plot")
            plot = create_plot(positions, scores, selected_scale_name, window_size, len(sequence))
            st.pyplot(plot)

            # --- Data Export ---
            st.header("3. Export Data")
            st.markdown("You can download the calculated hydrophobicity scores as a CSV file.")
            csv_data = "Position,Hydrophobicity_Score\n" + "\n".join([f"{p},{s:.3f}" for p, s in zip(positions, scores)])
            st.download_button(
                label="📥 Download Data as CSV",
                data=csv_data,
                file_name=f"hydrophobicity_{selected_scale_name.replace(' ', '_')}.csv",
                mime="text/csv",
            )


if __name__ == "__main__":
    main()
