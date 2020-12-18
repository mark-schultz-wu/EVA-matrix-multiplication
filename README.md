# Homomorphic Square Matrix Multiplication using the EVA optimizing Compiler

This is a quick implementation of the homomorphic square matrix multiplication
algorithm from the paper [Secure Outsourced Matrix Computation
and Application to Neural Networks](https://eprint.iacr.org/2018/1041.pdf) using
the the Python bindings for the [EVA optimizing compiler](https://github.com/microsoft/EVA).
This compiler uses the CKKS implementation within Microsoft SEAL's FHE library.

The EVA compiler is still quite new at the time of posting this, 
so I do not know how stable the Python bindings will be.

# Running

After installing EVA, you can test the code by running:

```python3 matrix_mul.py```

It should be configured to generate two random 64 x 64 matrices (with entries in
{0,...,5} uniformly), and multiply them. I print out results including:

1. The generated matrices
2. The result of the plaintext computation
3. The result of the matrix multiplication using a standard Python library
   (numpy)
4. The result of the FHE computation

Because of the size of the matrices (64 x 64) the entire matrices do not display on the
screen. You could simply edit the value `d` to something smaller.
If you want to investigate the (large) matrices in depth, run:

```python3 -i matrix_mul.py```

The generated plaintext matrices are saved as `matA` and `matB`.
The (decrypted) result of the homomorphic matrix multiplication is saved as
`matAB`.

# Tips for programming in EVA

At the time of writing, the EVA compiler is still quite new, and not nearly as
well-documented as Microsoft SEAL. Currently, it seems that the easiest way to
learn to use the EVA compiler is to look at valid EVA programs (which is part of
the motivation for posting this one).

Other sources of EVA programs that helped me are Microsoft's [image processing
example](https://github.com/microsoft/EVA/blob/main/examples/image_processing.py),
and the programs in their [tests
directory](https://github.com/microsoft/EVA/tree/main/tests).
