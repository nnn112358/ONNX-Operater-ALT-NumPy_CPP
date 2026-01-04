import numpy as np

def greaterorequal(A, B):
    """
    ONNX GreaterOrEqual オペレータ

    要素ごとの以上比較を行う。
    ブロードキャストをサポート。

    Args:
        A: 入力テンソル1
        B: 入力テンソル2

    Returns:
        C: A >= B の結果（ブール配列）
    """
    return np.greater_equal(A, B)


if __name__ == "__main__":
    # テスト例
    A = np.array([[1, 5, 3], [4, 2, 6]])
    B = np.array([[2, 3, 3], [4, 5, 1]])
    print("A:")
    print(A)
    print("\nB:")
    print(B)

    C = greaterorequal(A, B)
    print("\nGreaterOrEqual (A >= B):")
    print(C)

    # ブロードキャストの例
    A = np.array([[1, 2, 3], [4, 5, 6]])
    B = 3
    print("\n\nブロードキャスト例:")
    print("A:")
    print(A)
    print("B:", B)
    C = greaterorequal(A, B)
    print("GreaterOrEqual (A >= B):")
    print(C)

    # 等しい値の確認
    A = np.array([1.0, 2.0, 3.0])
    B = np.array([1.0, 1.5, 3.5])
    print("\n等値を含む比較:")
    print("A:", A)
    print("B:", B)
    print("GreaterOrEqual:", greaterorequal(A, B))
