import numpy as np

def reducesum(data, axes=None, keepdims=True):
    """
    ONNX ReduceSum オペレータ

    指定された軸に沿って合計を計算する。

    Args:
        data: 入力テンソル
        axes: 削減する軸のリスト (省略時は全軸)
        keepdims: 次元を保持するか (デフォルト: True)

    Returns:
        reduced: 合計値
    """
    if axes is not None:
        axes = tuple(axes) if isinstance(axes, list) else axes
    return np.sum(data, axis=axes, keepdims=keepdims)


if __name__ == "__main__":
    # テスト例
    data = np.array([[1, 2, 3], [4, 5, 6]])
    print("入力:")
    print(data)

    # 全要素の合計
    result = reducesum(data, axes=None)
    print("\nReduceSum (全軸):", result)

    # 軸0に沿った合計
    result = reducesum(data, axes=[0], keepdims=True)
    print("\nReduceSum (axis=0, keepdims=True):")
    print(result)

    # 軸1に沿った合計
    result = reducesum(data, axes=[1], keepdims=True)
    print("\nReduceSum (axis=1, keepdims=True):")
    print(result)

    # keepdims=False
    result = reducesum(data, axes=[1], keepdims=False)
    print("\nReduceSum (axis=1, keepdims=False):")
    print(result)

    # 3D配列の例
    data = np.arange(24).reshape(2, 3, 4)
    result = reducesum(data, axes=[1, 2], keepdims=True)
    print("\n3D配列 (axes=[1, 2]):")
    print("入力形状:", data.shape)
    print("出力形状:", result.shape)
