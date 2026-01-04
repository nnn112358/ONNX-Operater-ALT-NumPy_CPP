import numpy as np

def spacetodepth(input_tensor, blocksize):
    """
    ONNX SpaceToDepth オペレータ

    空間次元（H, W）をチャネル次元に再配置する。
    (N, C, H, W) -> (N, C * blocksize^2, H / blocksize, W / blocksize)

    Args:
        input_tensor: 入力テンソル (N, C, H, W)
        blocksize: ブロックサイズ

    Returns:
        output: 再配置されたテンソル
    """
    N, C, H, W = input_tensor.shape

    # 新しい形状
    new_H = H // blocksize
    new_W = W // blocksize
    new_C = C * (blocksize ** 2)

    # リシェイプと転置を使って実装
    # (N, C, H, W) -> (N, C, H//bs, bs, W//bs, bs)
    temp = input_tensor.reshape(N, C, new_H, blocksize, new_W, blocksize)

    # (N, C, H//bs, bs, W//bs, bs) -> (N, C, bs, bs, H//bs, W//bs)
    temp = temp.transpose(0, 1, 3, 5, 2, 4)

    # (N, C, bs, bs, H//bs, W//bs) -> (N, C*bs*bs, H//bs, W//bs)
    output = temp.reshape(N, new_C, new_H, new_W)

    return output


if __name__ == "__main__":
    # テスト例
    N, C, H, W = 1, 2, 4, 4
    blocksize = 2

    input_tensor = np.arange(N * C * H * W).reshape(N, C, H, W)
    print("入力形状:", input_tensor.shape)
    print("入力 (チャネル0):")
    print(input_tensor[0, 0])
    print("入力 (チャネル1):")
    print(input_tensor[0, 1])

    output = spacetodepth(input_tensor, blocksize)
    print("\nSpaceToDepth (blocksize=2):")
    print("出力形状:", output.shape)
    print("出力チャネル数:", output.shape[1])

    # より小さい例で確認
    input_small = np.arange(1 * 1 * 4 * 4).reshape(1, 1, 4, 4)
    print("\n\n小さい例:")
    print("入力形状:", input_small.shape)
    print("入力:")
    print(input_small[0, 0])

    output_small = spacetodepth(input_small, 2)
    print("\n出力形状:", output_small.shape)
    for i in range(output_small.shape[1]):
        print(f"\nチャネル{i}:")
        print(output_small[0, i])
