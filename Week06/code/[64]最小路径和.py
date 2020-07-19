# 给定一个包含非负整数的 m x n 网格，请找出一条从左上角到右下角的路径，使得路径上的数字总和为最小。 
# 
#  说明：每次只能向下或者向右移动一步。 
# 
#  示例: 
# 
#  输入:
# [
#   [1,3,1],
#   [1,5,1],
#   [4,2,1]
# ]
# 输出: 7
# 解释: 因为路径 1→3→1→1→1 的总和最小。
#  
#  Related Topics 数组 动态规划


# leetcode submit region begin(Prohibit modification and deletion)
class Solution:
    def minPathSum(self, grid: List[List[int]]) -> int:
        # dp数组：dp[i][j]记录从右下角到达[i, j]位置时的最小路径和
        if not grid or not grid[0]: return 0
        m, n = len(grid), len(grid[0])
        dp = [[0 for _ in range(n + 1)] for _ in range(m + 1)]
        for i in range(m - 1, -1, -1):
            for j in range(n - 1, -1, -1):
                if i == m - 1:
                    dp[i][j] = dp[i][j + 1] + grid[i][j]
                elif j == n - 1:
                    dp[i][j] = dp[i + 1][j] + grid[i][j]
                else:
                    dp[i][j] = min(dp[i + 1][j], dp[i][j + 1]) + grid[i][j]
        return dp[0][0]

# leetcode submit region end(Prohibit modification and deletion)
