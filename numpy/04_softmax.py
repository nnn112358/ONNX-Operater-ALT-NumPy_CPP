import numpy as np

def softmax(X, axis=-1):
    """
    ONNX Softmax オペレータ

    Softmax関数を適用する。
    数値安定性のため、最大値を引いてから計算する。

    Args:
        X: 入力テンソル
        axis: Softmaxを適用する軸 (デフォルト: -1)

    Returns:
        Y: Softmaxを適用した結果（確率分布）
    """
    # 数値安定性のため最大値を引く
    X_shifted = X - np.max(X, axis=axis, keepdims=True)

    # exp を計算
    exp_X = np.exp(X_shifted)

    # 正規化
    Y = exp_X / np.sum(exp_X, axis=axis, keepdims=True)

    return Y


if __name__ == "__main__":
    # テスト例1: 2D配列
    X = np.array([[1, 2, 3], [4, 5, 6]])
    print("入力:")
    print(X)

    Y = softmax(X, axis=-1)
    print("\nSoftmax出力 (axis=-1):")
    print(Y)
    print("各行の合計:", np.sum(Y, axis=-1))

    # テスト例2: 異なる軸
    Y = softmax(X, axis=0)
    print("\nSoftmax出力 (axis=0):")
    print(Y)
    print("各列の合計:", np.sum(Y, axis=0))

    # テスト例3: 分類問題の例
    logits = np.array([2.0, 1.0, 0.1])
    probs = softmax(logits)
    print("\nロジット:", logits)
    print("確率:", probs)
    print("合計:", np.sum(probs))
