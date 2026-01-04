import numpy as np

def transpose(data, perm=None):
    """
    ONNX Transpose オペレータ

    テンソルの次元を入れ替える。

    Args:
        data: 入力テンソル
        perm: 次元の順序を指定するリスト (省略時は逆順)

    Returns:
        transposed: 転置されたテンソル
    """
    if perm is None:
        return np.transpose(data)
    else:
        return np.transpose(data, perm)


if __name__ == "__main__":
    # テスト例 - 2D配列
    data = np.array([[1, 2, 3], [4, 5, 6]])
    print("元の形状:", data.shape)
    print(data)

    transposed = transpose(data)
    print("\nTranspose (デフォルト):")
    print("形状:", transposed.shape)
    print(transposed)

    # 3D配列での例
    data = np.arange(24).reshape(2, 3, 4)
    print("\n\n元の形状:", data.shape)

    # (2, 3, 4) -> (4, 2, 3)
    transposed = transpose(data, [2, 0, 1])
    print("Transpose with perm=[2, 0, 1]:")
    print("形状:", transposed.shape)
