from board import Renju
import numpy as  np

b = Renju()

action_mask, _, _, done = b.reset()
while True:
    print(action_mask.sum(axis=0))
    b.show()
    if done:
        break
    while True:
        i = list(map(int, (input(">>").split())))
        print(i)
        if len(i) != 2:
            print("サイズが違います")
            continue
        if i[0] < 0 or i[0] > 14 or i[1] < 0 or i[1] > 14:
            print("0~14の範囲で入力してください")
            continue
        if action_mask.sum(axis=0)[i[0], i[1]] == 0:
            print("既にコマが置いてあるか許可された位置ではありません")
            continue
        if True:
            break

    action_mask, _, _, done = b.step(i)

print(np.amax(b.board, axis=2))
