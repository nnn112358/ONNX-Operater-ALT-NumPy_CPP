import numpy as np

def sqrt(X):
    """
    ONNX Sqrt オペレータ

    テンソルの要素ごとに平方根を計算する。

    Args:
        X: 入力テンソル (非負の値)

    Returns:
        Y: sqrt(X) の結果
    """
    return np.sqrt(X)


if __name__ == "__main__":
    # テスト例
    X = np.array([[1, 4], [9, 16]])
    print("Sqrt(X) =")
    print(sqrt(X))

    # 浮動小数点数の例
    X = np.array([0, 1, 2, 3, 4, 5])
    print("\nSqrt(X) =")
    print(sqrt(X))
