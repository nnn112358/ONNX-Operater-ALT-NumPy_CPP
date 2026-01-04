import numpy as np

def log(X):
    """
    ONNX Log オペレータ

    テンソルの要素ごとに自然対数を計算する。

    Args:
        X: 入力テンソル (正の値)

    Returns:
        Y: log(X) の結果
    """
    return np.log(X)


if __name__ == "__main__":
    # テスト例
    X = np.array([[1, 2], [np.e, 10]])
    print("Log(X) =")
    print(log(X))

    # より多くの値の例
    X = np.array([1, 2, 3, 4, 5])
    print("\nLog(X) =")
    print(log(X))
