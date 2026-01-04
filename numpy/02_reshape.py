import numpy as np

def reshape(data, shape):
    """
    ONNX Reshape オペレータ

    テンソルを新しい形状に変形する。
    -1を使用すると、その次元は自動的に計算される。

    Args:
        data: 入力テンソル
        shape: 新しい形状を表すリストまたは配列

    Returns:
        reshaped: 変形されたテンソル
    """
    return np.reshape(data, shape)


if __name__ == "__main__":
    # テスト例
    data = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])
    print("元の形状:", data.shape)
    print(data)

    # (2, 4) -> (4, 2)
    reshaped = reshape(data, (4, 2))
    print("\nReshape to (4, 2):")
    print(reshaped)

    # (2, 4) -> (8,)
    reshaped = reshape(data, (8,))
    print("\nReshape to (8,):")
    print(reshaped)

    # -1 を使った自動計算
    reshaped = reshape(data, (-1, 1))
    print("\nReshape to (-1, 1):")
    print(reshaped)
