import numpy as np

def unsqueeze(data, axes):
    """
    ONNX Unsqueeze オペレータ

    指定した位置にサイズ1の新しい次元を追加する。

    Args:
        data: 入力テンソル
        axes: 追加する軸の位置のリスト

    Returns:
        unsqueezed: サイズ1の次元が追加されたテンソル
    """
    result = data
    # axesをソートして順番に適用
    for axis in sorted(axes):
        result = np.expand_dims(result, axis=axis)
    return result


if __name__ == "__main__":
    # テスト例
    data = np.array([[1, 2, 3], [4, 5, 6]])
    print("元の形状:", data.shape)
    print(data)

    # 軸0に次元を追加
    unsqueezed = unsqueeze(data, axes=[0])
    print("\nUnsqueeze (axes=[0]):")
    print("形状:", unsqueezed.shape)

    # 複数の軸に次元を追加
    unsqueezed = unsqueeze(data, axes=[0, 3])
    print("\nUnsqueeze (axes=[0, 3]):")
    print("形状:", unsqueezed.shape)

    # 1D配列の例
    data = np.array([1, 2, 3, 4])
    print("\n元の形状:", data.shape)
    unsqueezed = unsqueeze(data, axes=[0, 2])
    print("Unsqueeze (axes=[0, 2]):")
    print("形状:", unsqueezed.shape)
