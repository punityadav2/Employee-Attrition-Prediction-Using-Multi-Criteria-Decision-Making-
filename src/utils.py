"""
Utility functions for HR Analytics project
"""

import os
import matplotlib.pyplot as plt
import seaborn as sns


def set_plot_style():
    """Set consistent plotting style for all visualizations"""
    plt.style.use('seaborn-v0_8-darkgrid')
    sns.set_palette("Set2")
    plt.rcParams['figure.figsize'] = (10, 6)
    plt.rcParams['font.size'] = 10
    plt.rcParams['axes.labelsize'] = 11
    plt.rcParams['axes.titlesize'] = 12
    plt.rcParams['xtick.labelsize'] = 9
    plt.rcParams['ytick.labelsize'] = 9
    plt.rcParams['legend.fontsize'] = 10


def create_directories(dirs):
    """
    Create directories if they don't exist
    
    Args:
        dirs (list): List of directory paths to create
    """
    for directory in dirs:
        os.makedirs(directory, exist_ok=True)
    print(f"[OK] Created {len(dirs)} directories")


def log_message(message, level="INFO"):
    """
    Simple logging function
    
    Args:
        message (str): Message to log
        level (str): Log level (INFO, WARNING, ERROR)
    """
    prefix = {
        "INFO": "",
        "WARNING": "[!]",
        "ERROR": "[X]",
        "SUCCESS": "[OK]"
    }
    print(f"{prefix.get(level, '')} [{level}] {message}")


def print_section_header(title):
    """
    Print a formatted section header
    
    Args:
        title (str): Section title
    """
    print("\n" + "="*70)
    print(f"  {title.upper()}")
    print("="*70 + "\n")


def print_subsection_header(title):
    """
    Print a formatted subsection header
    
    Args:
        title (str): Subsection title
    """
    print("\n" + "-"*70)
    print(f"  {title}")
    print("-"*70)


if __name__ == "__main__":
    # Test utilities
    set_plot_style()
    log_message("Testing utils module", "SUCCESS")
    print_section_header("Test Section")
    print_subsection_header("Test Subsection")
