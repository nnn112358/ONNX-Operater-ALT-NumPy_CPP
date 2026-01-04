import numpy as np

def reducemax(data, axes=None, keepdims=True):
    """
    ONNX ReduceMax オペレータ

    指定された軸に沿って最大値を計算する。

    Args:
        data: 入力テンソル
        axes: 削減する軸のリスト (省略時は全軸)
        keepdims: 次元を保持するか (デフォルト: True)

    Returns:
        reduced: 最大値
    """
    if axes is not None:
        axes = tuple(axes) if isinstance(axes, list) else axes
    return np.max(data, axis=axes, keepdims=keepdims)


if __name__ == "__main__":
    # テスト例
    data = np.array([[1, 5, 3], [4, 2, 6]])
    print("入力:")
    print(data)

    # 全要素の最大値
    result = reducemax(data, axes=None)
    print("\nReduceMax (全軸):", result)

    # 軸0に沿った最大値
    result = reducemax(data, axes=[0], keepdims=True)
    print("\nReduceMax (axis=0, keepdims=True):")
    print(result)

    # 軸1に沿った最大値
    result = reducemax(data, axes=[1], keepdims=True)
    print("\nReduceMax (axis=1, keepdims=True):")
    print(result)

    # keepdims=False
    result = reducemax(data, axes=[1], keepdims=False)
    print("\nReduceMax (axis=1, keepdims=False):")
    print(result)

    # 3D配列の例
    data = np.random.randn(2, 3, 4)
    result = reducemax(data, axes=[1, 2], keepdims=True)
    print("\n3D配列 (axes=[1, 2]):")
    print("入力形状:", data.shape)
    print("出力形状:", result.shape)
