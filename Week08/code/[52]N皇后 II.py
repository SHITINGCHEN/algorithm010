# n 皇后问题研究的是如何将 n 个皇后放置在 n×n 的棋盘上，并且使皇后彼此之间不能相互攻击。 
# 
#  
# 
#  上图为 8 皇后问题的一种解法。 
# 
#  给定一个整数 n，返回 n 皇后不同的解决方案的数量。 
# 
#  示例: 
# 
#  输入: 4
# 输出: 2
# 解释: 4 皇后问题存在如下两个不同的解法。
# [
#  [".Q..",  // 解法 1
#   "...Q",
#   "Q...",
#   "..Q."],
# 
#  ["..Q.",  // 解法 2
#   "Q...",
#   "...Q",
#   ".Q.."]
# ]
#  
# 
#  
# 
#  提示： 
# 
#  
#  皇后，是国际象棋中的棋子，意味着国王的妻子。皇后只做一件事，那就是“吃子”。当她遇见可以吃的棋子时，就迅速冲上去吃掉棋子。当然，她横、竖、斜都可走一或 N
# -1 步，可进可退。（引用自 百度百科 - 皇后 ） 
#  
#  Related Topics 回溯算法


# leetcode submit region begin(Prohibit modification and deletion)
class Solution:
    def totalNQueens(self, n: int) -> int:
        if n < 1: return []
        self.count = 0
        self.dfs(n, 0, 0, 0, 0)
        return self.count

    def dfs(self, n, row, col, pie, na):
        if row == n:
            self.count += 1
            return
        # 记录当前有哪些空格可以放（1可以存，0不可以）
        # 后面与的部分，将超出n的位记为0
        bits = (~(col | pie | na)) & ((1 << n) - 1)
        while bits:
            p = bits & -bits # 取最低位的1
            bits = bits & (bits - 1) # 最低位的1变为0, 表示在p位置上放皇后
            # pie左移一位，因为往下一列这个撇往右斜下角多打一位，因此要左移
            # na同理
            self.dfs(n, row + 1, col | p, (pie | p) << 1, (na | p ) >> 1)



# leetcode submit region end(Prohibit modification and deletion)
