import numpy as np

def reducelogsum(data, axes=None, keepdims=True):
    """
    ONNX ReduceLogSum オペレータ

    指定された軸に沿ってlog(sum(x))を計算する。

    Args:
        data: 入力テンソル
        axes: 削減する軸のリスト (省略時は全軸)
        keepdims: 次元を保持するか (デフォルト: True)

    Returns:
        reduced: log(sum(x))
    """
    if axes is not None:
        axes = tuple(axes) if isinstance(axes, list) else axes
    sum_result = np.sum(data, axis=axes, keepdims=keepdims)
    return np.log(sum_result)


if __name__ == "__main__":
    # テスト例
    data = np.array([[1, 2, 3], [4, 5, 6]])
    print("入力:")
    print(data)

    # 全要素のLogSum
    result = reducelogsum(data, axes=None)
    print("\nReduceLogSum (全軸):", result)
    print("検証: log(1 + 2 + 3 + 4 + 5 + 6) = log(21) =", np.log(21))

    # 軸0に沿ったLogSum
    result = reducelogsum(data, axes=[0], keepdims=True)
    print("\nReduceLogSum (axis=0, keepdims=True):")
    print(result)

    # 軸1に沿ったLogSum
    result = reducelogsum(data, axes=[1], keepdims=True)
    print("\nReduceLogSum (axis=1, keepdims=True):")
    print(result)
    print("検証: [log(1+2+3), log(4+5+6)] = [log(6), log(15)] =", [np.log(6), np.log(15)])

    # keepdims=False
    result = reducelogsum(data, axes=[1], keepdims=False)
    print("\nReduceLogSum (axis=1, keepdims=False):")
    print(result)

    # 小数の例
    data = np.array([[0.5, 1.0], [1.5, 2.0]])
    result = reducelogsum(data, axes=[1], keepdims=True)
    print("\n小数での例:")
    print("入力:")
    print(data)
    print("ReduceLogSum (axis=1):")
    print(result)
