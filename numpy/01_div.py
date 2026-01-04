import numpy as np

def div(A, B):
    """
    ONNX Div オペレータ

    2つのテンソルの要素ごとの除算を行う。
    ブロードキャストをサポート。

    Args:
        A: 被除数テンソル
        B: 除数テンソル

    Returns:
        C: A / B の結果
    """
    return np.divide(A, B)


if __name__ == "__main__":
    # テスト例
    A = np.array([[10, 20], [30, 40]])
    B = np.array([[2, 4], [5, 8]])
    print("A / B =")
    print(div(A, B))

    # ブロードキャストの例
    A = np.array([[100, 200, 300]])
    B = np.array([[10]])
    print("\nブロードキャスト: A / B =")
    print(div(A, B))
