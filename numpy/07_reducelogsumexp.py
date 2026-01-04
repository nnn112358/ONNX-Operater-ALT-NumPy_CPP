import numpy as np

def reducelogsumexp(data, axes=None, keepdims=True):
    """
    ONNX ReduceLogSumExp オペレータ

    指定された軸に沿ってlog(sum(exp(x)))を計算する。
    数値安定性のため、最大値を引いてから計算する。

    Args:
        data: 入力テンソル
        axes: 削減する軸のリスト (省略時は全軸)
        keepdims: 次元を保持するか (デフォルト: True)

    Returns:
        reduced: log(sum(exp(x)))
    """
    if axes is not None:
        axes = tuple(axes) if isinstance(axes, list) else axes

    # 数値安定性のため最大値を引く
    max_val = np.max(data, axis=axes, keepdims=True)
    exp_shifted = np.exp(data - max_val)
    sum_exp = np.sum(exp_shifted, axis=axes, keepdims=keepdims)

    if keepdims:
        result = np.log(sum_exp) + max_val
    else:
        max_val_squeezed = np.squeeze(max_val, axis=axes)
        result = np.log(sum_exp) + max_val_squeezed

    return result


if __name__ == "__main__":
    # テスト例
    data = np.array([[1, 2, 3], [4, 5, 6]])
    print("入力:")
    print(data)

    # 全要素のLogSumExp
    result = reducelogsumexp(data, axes=None)
    print("\nReduceLogSumExp (全軸):", result)

    # 軸0に沿ったLogSumExp
    result = reducelogsumexp(data, axes=[0], keepdims=True)
    print("\nReduceLogSumExp (axis=0, keepdims=True):")
    print(result)

    # 軸1に沿ったLogSumExp
    result = reducelogsumexp(data, axes=[1], keepdims=True)
    print("\nReduceLogSumExp (axis=1, keepdims=True):")
    print(result)

    # keepdims=False
    result = reducelogsumexp(data, axes=[1], keepdims=False)
    print("\nReduceLogSumExp (axis=1, keepdims=False):")
    print(result)

    # 検証（小さい数値で）
    data_small = np.array([1.0, 2.0, 3.0])
    result = reducelogsumexp(data_small, axes=None)
    expected = np.log(np.exp(1) + np.exp(2) + np.exp(3))
    print(f"\n検証: LogSumExp([1, 2, 3]) = {result:.6f}")
    print(f"期待値: log(e^1 + e^2 + e^3) = {expected:.6f}")
