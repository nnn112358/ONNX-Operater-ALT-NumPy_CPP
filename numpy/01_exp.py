import numpy as np

def exp(X):
    """
    ONNX Exp オペレータ

    テンソルの要素ごとに自然指数関数 e^x を計算する。

    Args:
        X: 入力テンソル

    Returns:
        Y: e^X の結果
    """
    return np.exp(X)


if __name__ == "__main__":
    # テスト例
    X = np.array([[0, 1], [2, 3]])
    print("Exp(X) =")
    print(exp(X))

    # 負の値の例
    X = np.array([-1, 0, 1, 2])
    print("\nExp(X) =")
    print(exp(X))
