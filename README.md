# Image Deblurring

**Using matrix decomposition to deblur images** — DDA3005 Numerical Linear Algebra project, Kaggle competition **6th place**.

This project implements and compares numerical methods for image deblurring by solving the linear matrix equation **A_ℓ X A_r = B**, where **B** is the blurred image, **A_ℓ** and **A_r** are blurring kernels (Toeplitz-type), and **X** is the unknown sharp image.

---

## Features

- **Blurring kernel construction** — Two configurations: motion-type (Config1) and symmetric banded (Config2)
- **Multiple deblurring methods:**
  - **LU factorization** — Direct solve via `scipy.linalg.lu_factor` / `lu_solve`
  - **QR factorization (SciPy)** — Orthogonal-triangular decomposition
  - **Custom Householder QR** — Self-implemented QR for comparison
  - **Least-squares / regularized** — Pseudo-inverse and ridge regression (λ‖X‖²_F)
  - **Padding** — Border padding to improve deblurring quality
- **Conditioning analysis** — Kernel condition numbers and rank for Config1 vs Config2
- **Metrics** — Relative error, PSNR, runtime, and visual comparisons

---

## Project Structure

```
.
├── task_a_create_blur.py    # Build blur kernels A_ℓ, A_r and generate blurred images B
├── task_b_deblur.py         # Deblur with LU and QR (SciPy)
├── task_c_householder_qr.py # Custom Householder QR factorization
├── task_d_deblur_custom_qr.py # Deblur using custom QR from task_c
├── task_e_improvements.py   # Least-squares and regularized deblurring
├── task_f_padding.py        # Padding-based deblurring
├── demo_deblurring.py       # Demo script
├── test_images/             # Original test images (PNG)
├── DDA3005_Final_Report.md  # Full project report
└── README.md
```

---

## Requirements

- Python 3.7+
- NumPy  
- SciPy  
- Matplotlib  
- Pillow (PIL)

```bash
pip install numpy scipy matplotlib pillow
```

---

## Quick Start

1. **Generate blur kernels and blurred images** (Task A):

   ```bash
   python task_a_create_blur.py
   ```

2. **Run deblurring with LU and QR** (Task B):

   ```bash
   python task_b_deblur.py
   ```

3. **Compare custom Householder QR deblurring** (Task D):

   ```bash
   python task_d_deblur_custom_qr.py
   ```

4. **Run regularized / least-squares deblurring** (Task E):

   ```bash
   python task_e_improvements.py
   ```

5. **Padding-based deblurring** (Task F):

   ```bash
   python task_f_padding.py
   ```

Or run the demo:

```bash
python demo_deblurring.py
```

---

## Mathematical Overview

The blur model is **B = A_ℓ X A_r**. Deblurring solves for **X** in two steps:

1. Solve **A_ℓ Y = B** → **Y = A_ℓ⁻¹ B**
2. Solve **Y A_r = X A_r** → **X^T = (A_r^T)⁻¹ Y^T**

Each step uses a matrix factorization (LU or QR) of **A_ℓ** and **A_r^T**, then triangular (or least-squares) solves. For ill-conditioned kernels (Config1), least-squares and regularization (Task E) give more stable results than direct LU/QR.

---

## Report

Detailed methodology, conditioning analysis, and result figures are in **[DDA3005_Final_Report.md](DDA3005_Final_Report.md)** (and the compact version [DDA3005_Final_Report_Compact.md](DDA3005_Final_Report_Compact.md)).

---

## Author

Xiaoxia Sheng — DDA3005 Numerical Linear Algebra, Image Deblurring & QR Factorizations project.

---

## License

This repository is for educational use (course project).
