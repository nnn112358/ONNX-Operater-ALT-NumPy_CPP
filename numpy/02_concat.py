import numpy as np

def concat(tensors, axis=0):
    """
    ONNX Concat オペレータ

    複数のテンソルを指定された軸に沿って連結する。

    Args:
        tensors: 連結するテンソルのリスト
        axis: 連結する軸 (デフォルト: 0)

    Returns:
        result: 連結されたテンソル
    """
    return np.concatenate(tensors, axis=axis)


if __name__ == "__main__":
    # テスト例
    A = np.array([[1, 2], [3, 4]])
    B = np.array([[5, 6], [7, 8]])
    C = np.array([[9, 10], [11, 12]])

    print("A:")
    print(A)
    print("\nB:")
    print(B)
    print("\nC:")
    print(C)

    # 軸0に沿って連結
    result = concat([A, B, C], axis=0)
    print("\nConcat (axis=0):")
    print("形状:", result.shape)
    print(result)

    # 軸1に沿って連結
    result = concat([A, B, C], axis=1)
    print("\nConcat (axis=1):")
    print("形状:", result.shape)
    print(result)
