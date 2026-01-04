import numpy as np

def reduceprod(data, axes=None, keepdims=True):
    """
    ONNX ReduceProd オペレータ

    指定された軸に沿って積を計算する。

    Args:
        data: 入力テンソル
        axes: 削減する軸のリスト (省略時は全軸)
        keepdims: 次元を保持するか (デフォルト: True)

    Returns:
        reduced: 積
    """
    if axes is not None:
        axes = tuple(axes) if isinstance(axes, list) else axes
    return np.prod(data, axis=axes, keepdims=keepdims)


if __name__ == "__main__":
    # テスト例
    data = np.array([[1, 2, 3], [4, 5, 6]])
    print("入力:")
    print(data)

    # 全要素の積
    result = reduceprod(data, axes=None)
    print("\nReduceProd (全軸):", result)

    # 軸0に沿った積
    result = reduceprod(data, axes=[0], keepdims=True)
    print("\nReduceProd (axis=0, keepdims=True):")
    print(result)

    # 軸1に沿った積
    result = reduceprod(data, axes=[1], keepdims=True)
    print("\nReduceProd (axis=1, keepdims=True):")
    print(result)

    # keepdims=False
    result = reduceprod(data, axes=[1], keepdims=False)
    print("\nReduceProd (axis=1, keepdims=False):")
    print(result)

    # より小さい数値での例
    data = np.array([[1, 2], [3, 4]])
    result = reduceprod(data, axes=[1], keepdims=True)
    print("\n小さい数値での例:")
    print("入力:")
    print(data)
    print("ReduceProd (axis=1):")
    print(result)
