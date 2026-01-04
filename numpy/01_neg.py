import numpy as np

def neg(X):
    """
    ONNX Neg オペレータ

    テンソルの要素ごとに符号を反転する。

    Args:
        X: 入力テンソル

    Returns:
        Y: -X の結果
    """
    return np.negative(X)


if __name__ == "__main__":
    # テスト例
    X = np.array([[1, -2], [3, -4]])
    print("Neg(X) =")
    print(neg(X))

    # 浮動小数点数の例
    X = np.array([1.5, -2.5, 3.0])
    print("\nNeg(X) =")
    print(neg(X))
