# Cell Inspection Project

This project is designed to perform structure tensor analysis on grayscale images to compute gradients, energy, orientation, and coherency. It uses various gradient computation methods, including finite difference, Gaussian smoothing, cubic spline interpolation, Fourier transform, Riesz transform, and Hessian-based second-order derivatives.

## Features

- **Gradient Computation**: Multiple methods for computing image gradients.
- **Structure Tensor Analysis**: Computes energy, orientation, and coherency from the structure tensor.
- **OpenCV Integration**: Utilizes OpenCV for image processing and visualization.
- **Boost Integration**: Uses Boost for cubic spline interpolation.

## Dependencies

- **OpenCV**: Required for image processing and visualization.
- **Boost**: Required for cubic spline interpolation.

## Building the Project

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/CellInspection.git
   cd CellInspection