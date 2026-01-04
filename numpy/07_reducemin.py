import numpy as np

def reducemin(data, axes=None, keepdims=True):
    """
    ONNX ReduceMin オペレータ

    指定された軸に沿って最小値を計算する。

    Args:
        data: 入力テンソル
        axes: 削減する軸のリスト (省略時は全軸)
        keepdims: 次元を保持するか (デフォルト: True)

    Returns:
        reduced: 最小値
    """
    if axes is not None:
        axes = tuple(axes) if isinstance(axes, list) else axes
    return np.min(data, axis=axes, keepdims=keepdims)


if __name__ == "__main__":
    # テスト例
    data = np.array([[1, 5, 3], [4, 2, 6]])
    print("入力:")
    print(data)

    # 全要素の最小値
    result = reducemin(data, axes=None)
    print("\nReduceMin (全軸):", result)

    # 軸0に沿った最小値
    result = reducemin(data, axes=[0], keepdims=True)
    print("\nReduceMin (axis=0, keepdims=True):")
    print(result)

    # 軸1に沿った最小値
    result = reducemin(data, axes=[1], keepdims=True)
    print("\nReduceMin (axis=1, keepdims=True):")
    print(result)

    # keepdims=False
    result = reducemin(data, axes=[1], keepdims=False)
    print("\nReduceMin (axis=1, keepdims=False):")
    print(result)

    # 3D配列の例
    data = np.random.randn(2, 3, 4)
    result = reducemin(data, axes=[1, 2], keepdims=True)
    print("\n3D配列 (axes=[1, 2]):")
    print("入力形状:", data.shape)
    print("出力形状:", result.shape)
