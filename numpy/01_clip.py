import numpy as np

def clip(X, min_val=None, max_val=None):
    """
    ONNX Clip オペレータ

    テンソルの値を指定された範囲にクリップする。

    Args:
        X: 入力テンソル
        min_val: 最小値 (省略可能)
        max_val: 最大値 (省略可能)

    Returns:
        Y: クリップされた結果
    """
    return np.clip(X, min_val, max_val)


if __name__ == "__main__":
    # テスト例
    X = np.array([[-5, 0, 5], [10, 15, 20]])
    print("Clip(X, min=0, max=10) =")
    print(clip(X, 0, 10))

    # 最小値のみの例
    X = np.array([-3, -2, -1, 0, 1, 2, 3])
    print("\nClip(X, min=0) =")
    print(clip(X, min_val=0))

    # 最大値のみの例
    print("\nClip(X, max=2) =")
    print(clip(X, max_val=2))
