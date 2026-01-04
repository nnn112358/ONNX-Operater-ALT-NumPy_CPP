import numpy as np

def pow(X, Y):
    """
    ONNX Pow オペレータ

    2つのテンソルの要素ごとのべき乗を計算する。
    X^Y を計算。ブロードキャストをサポート。

    Args:
        X: 底テンソル
        Y: 指数テンソル

    Returns:
        Z: X^Y の結果
    """
    return np.power(X, Y)


if __name__ == "__main__":
    # テスト例
    X = np.array([[2, 3], [4, 5]])
    Y = np.array([[2, 2], [3, 2]])
    print("X^Y =")
    print(pow(X, Y))

    # ブロードキャストの例
    X = np.array([2, 3, 4])
    Y = 2
    print("\nX^2 =")
    print(pow(X, Y))
