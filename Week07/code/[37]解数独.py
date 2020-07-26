# 编写一个程序，通过已填充的空格来解决数独问题。 
# 
#  一个数独的解法需遵循如下规则： 
# 
#  
#  数字 1-9 在每一行只能出现一次。 
#  数字 1-9 在每一列只能出现一次。 
#  数字 1-9 在每一个以粗实线分隔的 3x3 宫内只能出现一次。 
#  
# 
#  空白格用 '.' 表示。 
# 
#  
# 
#  一个数独。 
# 
#  
# 
#  答案被标成红色。 
# 
#  Note: 
# 
#  
#  给定的数独序列只包含数字 1-9 和字符 '.' 。 
#  你可以假设给定的数独只有唯一解。 
#  给定数独永远是 9x9 形式的。 
#  
#  Related Topics 哈希表 回溯算法


# leetcode submit region begin(Prohibit modification and deletion)
# import heapq
# from collections import namedtuple
class Solution:
    def solveSudoku(self, board: List[List[str]]) -> None:
        """
        Do not return anything, modify board in-place instead.
        """
        rows = [set(range(1, 10)) for _ in range(9)] # 初始化可以用来填空的数字
        cols = [set(range(1, 10)) for _ in range(9)]
        boxes = [set(range(1, 10)) for _ in range(9)]

        empty = [] # 记录需要填空的位置
        for i in range(9):
            for j in range(9):
                if board[i][j] != '.':
                    num = int(board[i][j])
                    rows[i].remove(num)
                    cols[j].remove(num)
                    boxes[i // 3 * 3 + j // 3].remove(num)
                else:
                    empty.append((i, j))



        def backtrack(index=0):
            if index == len(empty):
                return True
            i, j = empty[index]
            b = i // 3 * 3 + j // 3
            for num in rows[i] & cols[j] & boxes[b]:
                rows[i].remove(num)
                cols[j].remove(num)
                boxes[b].remove(num)
                board[i][j] = str(num)
                if backtrack(index + 1):
                    return True
                rows[i].add(num)
                cols[j].add(num)
                boxes[b].add(num)
            return False

        backtrack(0)

# leetcode submit region end(Prohibit modification and deletion)
