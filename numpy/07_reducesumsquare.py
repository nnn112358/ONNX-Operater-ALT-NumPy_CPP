import numpy as np

def reducesumsquare(data, axes=None, keepdims=True):
    """
    ONNX ReduceSumSquare オペレータ

    指定された軸に沿って二乗和を計算する。
    SumSquare = sum(x^2)

    Args:
        data: 入力テンソル
        axes: 削減する軸のリスト (省略時は全軸)
        keepdims: 次元を保持するか (デフォルト: True)

    Returns:
        reduced: 二乗和
    """
    if axes is not None:
        axes = tuple(axes) if isinstance(axes, list) else axes
    return np.sum(data ** 2, axis=axes, keepdims=keepdims)


if __name__ == "__main__":
    # テスト例
    data = np.array([[1, 2, 3], [4, 5, 6]])
    print("入力:")
    print(data)

    # 全要素の二乗和
    result = reducesumsquare(data, axes=None)
    print("\nReduceSumSquare (全軸):", result)
    print("検証: 1^2 + 2^2 + 3^2 + 4^2 + 5^2 + 6^2 =", 1 + 4 + 9 + 16 + 25 + 36)

    # 軸0に沿った二乗和
    result = reducesumsquare(data, axes=[0], keepdims=True)
    print("\nReduceSumSquare (axis=0, keepdims=True):")
    print(result)

    # 軸1に沿った二乗和
    result = reducesumsquare(data, axes=[1], keepdims=True)
    print("\nReduceSumSquare (axis=1, keepdims=True):")
    print(result)
    print("検証: [1^2 + 2^2 + 3^2, 4^2 + 5^2 + 6^2] =", [1 + 4 + 9, 16 + 25 + 36])

    # keepdims=False
    result = reducesumsquare(data, axes=[1], keepdims=False)
    print("\nReduceSumSquare (axis=1, keepdims=False):")
    print(result)
