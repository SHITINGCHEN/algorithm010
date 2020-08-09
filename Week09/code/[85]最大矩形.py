# 给定一个仅包含 0 和 1 的二维二进制矩阵，找出只包含 1 的最大矩形，并返回其面积。 
# 
#  示例: 
# 
#  输入:
# [
#   ["1","0","1","0","0"],
#   ["1","0","1","1","1"],
#   ["1","1","1","1","1"],
#   ["1","0","0","1","0"]
# ]
# 输出: 6 
#  Related Topics 栈 数组 哈希表 动态规划


# leetcode submit region begin(Prohibit modification and deletion)
class Solution:
    def maximalRectangle(self, matrix: List[List[str]]) -> int:
        if not matrix: return 0
        m, n = len(matrix), len(matrix[0])
        left = [0] * n
        right = [n] * n
        height = [0] * n
        max_area = 0
        for i in range(m):
            cur_left, cur_right = 0, n # 初始化当前最远的左边界和右边界
            for j in range(n):
                if matrix[i][j] == '1':
                    height[j] += 1
                else:
                    height[j] = 0
            for j in range(n):
                if matrix[i][j] == '1':
                    left[j] = max(left[j], cur_left)
                else:
                    left[j] = 0
                    cur_left = j + 1
            for j in range(n - 1, -1, -1):
                if matrix[i][j] == '1':
                    right[j] = min(right[j], cur_right)
                else:
                    right[j] = n
                    cur_right = j
            for j in range(n):
                max_area = max(max_area, height[j] * (right[j] - left[j]))

        return max_area
# leetcode submit region end(Prohibit modification and deletion)
