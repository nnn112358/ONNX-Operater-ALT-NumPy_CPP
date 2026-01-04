import numpy as np

def gemm(A, B, C=None, alpha=1.0, beta=1.0, transA=False, transB=False):
    """
    ONNX Gemm オペレータ

    一般行列乗算（General Matrix Multiplication）。
    Y = alpha * A' * B' + beta * C
    A', B' はそれぞれ転置オプションに従う。

    Args:
        A: 入力行列 (M, K) または (K, M) if transA
        B: 入力行列 (K, N) または (N, K) if transB
        C: バイアス行列 (省略可能、ブロードキャスト可能)
        alpha: A*Bのスカラー倍数 (デフォルト: 1.0)
        beta: Cのスカラー倍数 (デフォルト: 1.0)
        transA: Aを転置するか (デフォルト: False)
        transB: Bを転置するか (デフォルト: False)

    Returns:
        Y: 結果行列
    """
    # 転置処理
    A_mat = A.T if transA else A
    B_mat = B.T if transB else B

    # 行列乗算
    Y = alpha * np.matmul(A_mat, B_mat)

    # バイアス項の追加
    if C is not None:
        Y = Y + beta * C

    return Y


if __name__ == "__main__":
    # テスト例1: 基本的なGEMM
    A = np.array([[1, 2, 3], [4, 5, 6]])  # (2, 3)
    B = np.array([[7, 8], [9, 10], [11, 12]])  # (3, 2)
    C = np.array([[1, 1], [1, 1]])  # (2, 2)

    print("A:")
    print(A)
    print("\nB:")
    print(B)
    print("\nC:")
    print(C)

    Y = gemm(A, B, C, alpha=1.0, beta=1.0)
    print("\nGEMM (Y = A*B + C):")
    print(Y)

    # テスト例2: 転置あり
    A = np.array([[1, 2], [3, 4], [5, 6]])  # (3, 2)
    B = np.array([[7, 8, 9], [10, 11, 12]])  # (2, 3)

    Y = gemm(A, B, transA=True, transB=False)
    print("\n\nGEMM with transA=True:")
    print(f"A形状: {A.shape}, A^T形状: {A.T.shape}")
    print(f"B形状: {B.shape}")
    print(f"Y形状: {Y.shape}")
    print("Y:")
    print(Y)

    # テスト例3: スカラー倍数
    A = np.array([[1, 2], [3, 4]])
    B = np.array([[5, 6], [7, 8]])
    C = np.array([[10, 10], [10, 10]])

    Y = gemm(A, B, C, alpha=2.0, beta=0.5)
    print("\n\nGEMM (Y = 2*A*B + 0.5*C):")
    print(Y)
