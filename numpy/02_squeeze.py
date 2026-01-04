import numpy as np

def squeeze(data, axes=None):
    """
    ONNX Squeeze オペレータ

    サイズが1の次元を削除する。

    Args:
        data: 入力テンソル
        axes: 削除する軸のリスト (省略時は全てのサイズ1の次元を削除)

    Returns:
        squeezed: サイズ1の次元が削除されたテンソル
    """
    if axes is None:
        return np.squeeze(data)
    else:
        return np.squeeze(data, axis=tuple(axes) if isinstance(axes, list) else axes)


if __name__ == "__main__":
    # テスト例
    data = np.ones((1, 3, 1, 5))
    print("元の形状:", data.shape)

    # 全てのサイズ1の次元を削除
    squeezed = squeeze(data)
    print("Squeeze (全て):")
    print("形状:", squeezed.shape)

    # 特定の軸のみ削除
    squeezed = squeeze(data, axes=[0])
    print("\nSqueeze (axes=[0]):")
    print("形状:", squeezed.shape)

    squeezed = squeeze(data, axes=[0, 2])
    print("\nSqueeze (axes=[0, 2]):")
    print("形状:", squeezed.shape)
