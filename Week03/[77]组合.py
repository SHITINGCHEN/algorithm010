# 给定两个整数 n 和 k，返回 1 ... n 中所有可能的 k 个数的组合。 
# 
#  示例: 
# 
#  输入: n = 4, k = 2
# 输出:
# [
#   [2,4],
#   [3,4],
#   [2,3],
#   [1,2],
#   [1,3],
#   [1,4],
# ] 
#  Related Topics 回溯算法


# leetcode submit region begin(Prohibit modification and deletion)
class Solution:
    def combine(self, n: int, k: int) -> List[List[int]]:
        res = []
        if n < k or k <= 0 or n <= 0:
            return res
        def generate(cur, level):
            if len(cur) == k:
                res.append(cur)
                return
            for i in range(level, n + 1):
                generate(cur + [i], i + 1)

        generate([], 1)
        return res
# leetcode submit region end(Prohibit modification and deletion)
