# 给定一个可包含重复数字的序列，返回所有不重复的全排列。 
# 
#  示例: 
# 
#  输入: [1,1,2]
# 输出:
# [
#   [1,1,2],
#   [1,2,1],
#   [2,1,1]
# ] 
#  Related Topics 回溯算法


# leetcode submit region begin(Prohibit modification and deletion)
class Solution:
    def permuteUnique(self, nums: List[int]) -> List[List[int]]:
        res, size = [], len(nums)
        if size == 0:
            return res
        nums.sort()
        used = [False] * size
        def dfs(cur, depth):
            if depth == size:
                res.append(cur)
                return
            for i in range(size):
                if not used[i]:
                    if i > 0 and nums[i] == nums[i-1] and not used[i-1]:
                        continue
                    used[i] = True
                    dfs(cur + [nums[i]], depth + 1)
                    used[i] = False
        dfs([], 0)
        return res
# leetcode submit region end(Prohibit modification and deletion)
