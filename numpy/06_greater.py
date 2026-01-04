import numpy as np

def greater(A, B):
    """
    ONNX Greater オペレータ

    要素ごとの大なり比較を行う。
    ブロードキャストをサポート。

    Args:
        A: 入力テンソル1
        B: 入力テンソル2

    Returns:
        C: A > B の結果（ブール配列）
    """
    return np.greater(A, B)


if __name__ == "__main__":
    # テスト例
    A = np.array([[1, 5, 3], [4, 2, 6]])
    B = np.array([[2, 3, 3], [4, 5, 1]])
    print("A:")
    print(A)
    print("\nB:")
    print(B)

    C = greater(A, B)
    print("\nGreater (A > B):")
    print(C)

    # ブロードキャストの例
    A = np.array([[1, 2, 3], [4, 5, 6]])
    B = 3
    print("\n\nブロードキャスト例:")
    print("A:")
    print(A)
    print("B:", B)
    C = greater(A, B)
    print("Greater (A > B):")
    print(C)

    # 浮動小数点数の例
    A = np.array([1.5, 2.5, 3.5])
    B = np.array([1.0, 3.0, 3.5])
    print("\n浮動小数点数の比較:")
    print("A:", A)
    print("B:", B)
    print("Greater:", greater(A, B))
