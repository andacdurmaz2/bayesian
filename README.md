# Bayes Project

## SRC
Data Import

    Purpose: Load and preprocess temperature data
    Key Features:

    Loads CSV with lon, year, mean columns
    Filters longitude range (30°-60°)
    Organizes data by longitude into dictionary

    Returns: {longitude: DataFrame with year & mean}

BSplines
    Purpose: Generate B-spline basis functions for flexible modeling
    Key Features:

    Creates uniform B-spline basis
    Customizable: time range, number of basis functions, degree
    Returns evaluation points and basis matrix
    Wrapper around scipy.interpolate.BSpline

## testing
BSpline_test.py

    Purpose: Tests B-spline basis function generation and visualization
    What it does:

    Creates B-spline basis with 8 cubic functions
    Tests with uniform and random coefficients
    Plots resulting spline curves and coefficients
    Verifies matrix shapes and calculations

    Run with: python -m testing.BSpline_test

data_import_test.py

    Purpose: Tests data loading and creates exploratory visualizations
    What it does:

    Loads temperature data from CSV
    Filters longitude range (30°-60°)
    Creates scatter plot of mean temperatures by longitude and year
    Uses color coding for different years

    Run with: python -m testing.data_import_test

### PIP
pip install numpy scipy matplotlib pandas



