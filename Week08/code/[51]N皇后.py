# n 皇后问题研究的是如何将 n 个皇后放置在 n×n 的棋盘上，并且使皇后彼此之间不能相互攻击。 
# 
#  
# 
#  上图为 8 皇后问题的一种解法。 
# 
#  给定一个整数 n，返回所有不同的 n 皇后问题的解决方案。 
# 
#  每一种解法包含一个明确的 n 皇后问题的棋子放置方案，该方案中 'Q' 和 '.' 分别代表了皇后和空位。 
# 
#  示例: 
# 
#  输入: 4
# 输出: [
#  [".Q..",  // 解法 1
#   "...Q",
#   "Q...",
#   "..Q."],
# 
#  ["..Q.",  // 解法 2
#   "Q...",
#   "...Q",
#   ".Q.."]
# ]
# 解释: 4 皇后问题存在两个不同的解法。
#  
# 
#  
# 
#  提示： 
# 
#  
#  皇后，是国际象棋中的棋子，意味着国王的妻子。皇后只做一件事，那就是“吃子”。当她遇见可以吃的棋子时，就迅速冲上去吃掉棋子。当然，她横、竖、斜都可走一到七步
# ，可进可退。（引用自 百度百科 - 皇后 ） 
#  
#  Related Topics 回溯算法


# leetcode submit region begin(Prohibit modification and deletion)
class Solution:
    def solveNQueens(self, n: int) -> List[List[str]]:
        if n < 1: return []
        self.res = []
        self.dfs(n, [], 0, 0, 0)
        # return self.res
        return [['.' * (i - 1) + 'Q' + '.' * (n - i) for i in res] for res in self.res]

    def dfs(self, n, row, col, pie, na):
        if len(row) == n:
            self.res.append([self.log2(p) for p in row])
            return
        # 记录当前有哪些空格可以放（1可以存，0不可以）
        # 后面与的部分，将超出n的位记为0
        bits = (~(col | pie | na)) & ((1 << n) - 1)
        while bits:
            p = bits & -bits # 取低位的1
            bits = bits & (bits - 1) # 表示在p位置放皇后
            self.dfs(n, row + [p], col | p, (pie | p) << 1, (na | p) >> 1)

    def log2(self, p):
        count = 0
        while p:
            count += 1
            p >>= 1
        return count

# leetcode submit region end(Prohibit modification and deletion)
