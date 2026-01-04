import numpy as np

def add(A, B):
    """
    ONNX Add オペレータ

    2つのテンソルの要素ごとの加算を行う。
    ブロードキャストをサポート。

    Args:
        A: 入力テンソル1
        B: 入力テンソル2

    Returns:
        C: A + B の結果
    """
    return np.add(A, B)


if __name__ == "__main__":
    # テスト例
    A = np.array([[1, 2], [3, 4]])
    B = np.array([[5, 6], [7, 8]])
    print("A + B =")
    print(add(A, B))

    # ブロードキャストの例
    A = np.array([[1, 2, 3], [4, 5, 6]])
    B = np.array([10, 20, 30])
    print("\nブロードキャスト: A + B =")
    print(add(A, B))
