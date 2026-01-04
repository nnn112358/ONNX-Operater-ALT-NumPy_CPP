import numpy as np

def sub(A, B):
    """
    ONNX Sub オペレータ

    2つのテンソルの要素ごとの減算を行う。
    ブロードキャストをサポート。

    Args:
        A: 被減数テンソル
        B: 減数テンソル

    Returns:
        C: A - B の結果
    """
    return np.subtract(A, B)


if __name__ == "__main__":
    # テスト例
    A = np.array([[10, 20], [30, 40]])
    B = np.array([[1, 2], [3, 4]])
    print("A - B =")
    print(sub(A, B))

    # ブロードキャストの例
    A = np.array([[100, 200, 300]])
    B = np.array([50])
    print("\nブロードキャスト: A - B =")
    print(sub(A, B))
