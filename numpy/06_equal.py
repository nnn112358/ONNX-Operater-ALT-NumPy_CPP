import numpy as np

def equal(A, B):
    """
    ONNX Equal オペレータ

    要素ごとの等価比較を行う。
    ブロードキャストをサポート。

    Args:
        A: 入力テンソル1
        B: 入力テンソル2

    Returns:
        C: A == B の結果（ブール配列）
    """
    return np.equal(A, B)


if __name__ == "__main__":
    # テスト例
    A = np.array([[1, 2, 3], [4, 5, 6]])
    B = np.array([[1, 0, 3], [4, 5, 0]])
    print("A:")
    print(A)
    print("\nB:")
    print(B)

    C = equal(A, B)
    print("\nEqual (A == B):")
    print(C)

    # ブロードキャストの例
    A = np.array([[1, 2, 3], [1, 2, 3]])
    B = np.array([1, 2, 3])
    print("\n\nブロードキャスト例:")
    print("A:")
    print(A)
    print("B:", B)
    C = equal(A, B)
    print("Equal (A == B):")
    print(C)

    # 浮動小数点数の例
    A = np.array([1.0, 2.5, 3.7])
    B = np.array([1.0, 2.5, 3.8])
    print("\n浮動小数点数の比較:")
    print("A:", A)
    print("B:", B)
    print("Equal:", equal(A, B))
