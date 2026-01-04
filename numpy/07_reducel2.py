import numpy as np

def reducel2(data, axes=None, keepdims=True):
    """
    ONNX ReduceL2 オペレータ

    指定された軸に沿ってL2ノルムを計算する。
    L2 = sqrt(sum(x^2))

    Args:
        data: 入力テンソル
        axes: 削減する軸のリスト (省略時は全軸)
        keepdims: 次元を保持するか (デフォルト: True)

    Returns:
        reduced: L2ノルム
    """
    if axes is not None:
        axes = tuple(axes) if isinstance(axes, list) else axes
    return np.sqrt(np.sum(data ** 2, axis=axes, keepdims=keepdims))


if __name__ == "__main__":
    # テスト例
    data = np.array([[3, 4], [5, 12]])
    print("入力:")
    print(data)

    # 全要素のL2ノルム
    result = reducel2(data, axes=None)
    print("\nReduceL2 (全軸):", result)
    print("検証: sqrt(3^2 + 4^2 + 5^2 + 12^2) =", np.sqrt(9 + 16 + 25 + 144))

    # 軸0に沿ったL2ノルム
    result = reducel2(data, axes=[0], keepdims=True)
    print("\nReduceL2 (axis=0, keepdims=True):")
    print(result)

    # 軸1に沿ったL2ノルム
    result = reducel2(data, axes=[1], keepdims=True)
    print("\nReduceL2 (axis=1, keepdims=True):")
    print(result)
    print("検証: [sqrt(3^2 + 4^2), sqrt(5^2 + 12^2)] =", [np.sqrt(9 + 16), np.sqrt(25 + 144)])

    # keepdims=False
    result = reducel2(data, axes=[1], keepdims=False)
    print("\nReduceL2 (axis=1, keepdims=False):")
    print(result)
