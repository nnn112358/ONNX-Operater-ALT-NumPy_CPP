import numpy as np

def reversesequence(input_tensor, sequence_lens, batch_axis=1, time_axis=0):
    """
    ONNX ReverseSequence オペレータ

    各バッチについて、指定された長さまでシーケンスを反転する。

    Args:
        input_tensor: 入力テンソル
        sequence_lens: 各バッチの反転する長さ
        batch_axis: バッチ軸 (デフォルト: 1)
        time_axis: 時間軸（反転する軸） (デフォルト: 0)

    Returns:
        output: 反転されたテンソル
    """
    output = input_tensor.copy()

    # バッチ軸のサイズ
    batch_size = input_tensor.shape[batch_axis]

    for batch_idx in range(batch_size):
        seq_len = sequence_lens[batch_idx]

        # インデックスを作成
        indices = [slice(None)] * len(input_tensor.shape)
        indices[batch_axis] = batch_idx

        # 時間軸のスライス
        time_indices = [slice(None)] * len(input_tensor.shape)
        time_indices[batch_axis] = batch_idx
        time_indices[time_axis] = slice(0, seq_len)

        # 反転したデータ
        reversed_indices = [slice(None)] * len(input_tensor.shape)
        reversed_indices[batch_axis] = batch_idx
        reversed_indices[time_axis] = slice(seq_len - 1, None, -1)

        # データを反転
        output[tuple(time_indices)] = input_tensor[tuple(reversed_indices)]

    return output


if __name__ == "__main__":
    # テスト例1: 2D配列
    input_tensor = np.array([[1, 2, 3, 4],
                             [5, 6, 7, 8],
                             [9, 10, 11, 12]])
    sequence_lens = np.array([3, 2, 4])

    print("入力 (3x4):")
    print(input_tensor)
    print("\nシーケンス長:", sequence_lens)

    output = reversesequence(input_tensor, sequence_lens, batch_axis=0, time_axis=1)
    print("\nReverseSequence (batch_axis=0, time_axis=1):")
    print(output)
    print("\n説明:")
    print("行0: 最初の3要素を反転 [1,2,3,4] -> [3,2,1,4]")
    print("行1: 最初の2要素を反転 [5,6,7,8] -> [6,5,7,8]")
    print("行2: 最初の4要素を反転 [9,10,11,12] -> [12,11,10,9]")

    # テスト例2: 時系列データ
    print("\n\n時系列データの例:")
    input_tensor = np.array([[[1, 2], [3, 4], [5, 6]],
                             [[7, 8], [9, 10], [11, 12]]])
    sequence_lens = np.array([2, 3])

    print("入力形状:", input_tensor.shape, "(time, batch, features)")
    print("シーケンス長:", sequence_lens)

    output = reversesequence(input_tensor, sequence_lens, batch_axis=1, time_axis=0)
    print("\nReverseSequence (batch_axis=1, time_axis=0):")
    print("バッチ0:")
    print(output[:, 0, :])
    print("\nバッチ1:")
    print(output[:, 1, :])
