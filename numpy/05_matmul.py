import numpy as np

def matmul(A, B):
    """
    ONNX MatMul オペレータ

    行列乗算を行う。
    2D以上のテンソルに対応し、最後の2次元で行列乗算を実行。

    Args:
        A: 入力行列/テンソル
        B: 入力行列/テンソル

    Returns:
        C: A @ B の結果
    """
    return np.matmul(A, B)


if __name__ == "__main__":
    # テスト例1: 2D行列
    A = np.array([[1, 2], [3, 4]])
    B = np.array([[5, 6], [7, 8]])
    print("A:")
    print(A)
    print("\nB:")
    print(B)

    C = matmul(A, B)
    print("\nMatMul (A @ B):")
    print(C)

    # テスト例2: ベクトルと行列
    A = np.array([1, 2, 3])
    B = np.array([[1, 2], [3, 4], [5, 6]])
    print("\n\nベクトル A:", A)
    print("行列 B:")
    print(B)
    C = matmul(A, B)
    print("MatMul (A @ B):", C)

    # テスト例3: バッチ行列乗算
    A = np.random.randn(2, 3, 4)  # バッチサイズ2, 3x4行列
    B = np.random.randn(2, 4, 5)  # バッチサイズ2, 4x5行列
    C = matmul(A, B)
    print("\n\nバッチ行列乗算:")
    print(f"A形状: {A.shape}")
    print(f"B形状: {B.shape}")
    print(f"C形状: {C.shape}")
