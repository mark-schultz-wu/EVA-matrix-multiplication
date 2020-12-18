from eva.seal import *
from eva.ckks import *
from eva import *
from eva.metric import valuation_mse

import numpy as np
import random


def vec_from_pred(n, pred):
    # Returns a vector v in {0,1}^n s.t.
    # v[i] = pred(i)
    return [1 if pred(ell) else 0 for ell in range(n)]


def rot(vec, k):
    # Left-rotate the CKKS ciphertext by k slots
    return vec << k


def hom_matrix_mul(matA, matB):
    # Encrypts matA, matB, and then homomorphically multiplies them
    # using the homomorphic matrix multiplication algorithm from
    # https://eprint.iacr.org/2018/1041.pdf

    # Checking that A, B are both d x d matrices for some d
    # d will have to be a power of two for CKKS to work, I won't check here
    d = len(matA)
    assert(len(matB) == d)
    assert(all(len(matA[i]) == d for i in range(d)))
    assert(all(len(matB[i]) == d for i in range(d)))

    program = EvaProgram(
        f'Multiplication of two {d} x {d} matrices', vec_size=d*d)
    with program:
        n = d**2
        # Step 1-1
        a = Input('a')
        ctA0 = [0 for _ in range(n)]
        for k in range(-d-1, d):
            if k >= 0:
                uk = vec_from_pred(n, lambda ell: 0 <= ell - d * k < (d-k))
            else:
                uk = vec_from_pred(n, lambda ell: -k <= ell - (d+k) * d < d)
            ctA0 += rot(a, k) * uk
        # Step 1-2
        b = Input('b')
        ctB0 = [0 for _ in range(n)]
        for k in range(d):
            ctB0 += rot(b, d * k) * vec_from_pred(n,
                                                  lambda ell: ell % d == k)
        # Step 3
        ctAB = ctA0 * ctB0
        for k in range(1, d):
            # Step 2
            vk = vec_from_pred(n, lambda ell: 0 <= ell % d < d - k)
            vk_minus_d = vec_from_pred(n, lambda ell: (d-k) <= ell % d < d)
            ctAk = rot(ctA0, k) * vk + rot(ctA0, k-d) * vk_minus_d
            ctBk = rot(ctB0, d * k)

            # Second part of step 3
            ctAB += ctAk * ctBk

        Output('AB', ctAB)

    program.set_output_ranges(30)
    program.set_input_scales(30)

    compiler = CKKSCompiler()
    circuit, params, signature = compiler.compile(program)

    public_key, secret_key = generate_keys(params)

    # Vectorizing matrices
    a = [elem for row in matA for elem in row]
    b = [elem for row in matB for elem in row]
    inputs = {'a': a, 'b': b}

    encInputs = public_key.encrypt(inputs, signature)
    encOutputs = public_key.execute(circuit, encInputs)
    outputs = secret_key.decrypt(encOutputs, signature)
    reference = evaluate(circuit, inputs)
    print("Multiplying the matrices:")
    print(np.matrix(matA))
    print(np.matrix(matB))
    print("Expected result:")
    print(np.reshape(np.array(reference['AB']), (d, d)))
    print("Verified Expected result (Matrix mul using Numpy):")
    matA = np.array(matA)
    matB = np.array(matB)
    print(np.matmul(matA, matB))
    print("Actual result:")
    print(np.reshape(np.array(outputs['AB']), (d, d)))
    return outputs


if __name__ == "__main__":
    d = 64
    matA = [[random.randint(0, 5) for _ in range(d)] for _ in range(d)]
    matB = [[random.randint(0, 5) for _ in range(d)] for _ in range(d)]
    matAB = hom_matrix_mul(matA, matB)
