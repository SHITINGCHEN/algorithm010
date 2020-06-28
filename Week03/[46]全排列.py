# 给定一个 没有重复 数字的序列，返回其所有可能的全排列。 
# 
#  示例: 
# 
#  输入: [1,2,3]
# 输出:
# [
#   [1,2,3],
#   [1,3,2],
#   [2,1,3],
#   [2,3,1],
#   [3,1,2],
#   [3,2,1]
# ] 
#  Related Topics 回溯算法


# leetcode submit region begin(Prohibit modification and deletion)
class Solution:
    def permute(self, nums: List[int]) -> List[List[int]]:
        def dfs(cur, nums):
            if not nums:
                res.append(cur)
            for i in range(len(nums)):
                dfs(cur + [nums[i]], nums[:i] + nums[i+1:])
        res = []
        dfs([], nums)
        return res

        # res = []
        # index_set = set()
        # def generate(cur):
        #     if len(cur) == len(nums):
        #         res.append(cur)
        #         return
        #     for i in range(len(nums)):
        #         if i in index_set:
        #             continue
        #         index_set.add(i)
        #         generate(cur + [nums[i]])
        #         index_set.remove(i)
        # generate([])
        # return res

        # 不使用额外的数组标记状态
        # def backtrace(first = 0):
        #     if first == n:
        #         res.append(nums[:])
        #         return
        #     for i in range(first, n):
        #         nums[first], nums[i] = nums[i], nums[first]
        #         # drill down
        #         backtrace(first + 1)
        #         # reverse
        #         nums[first], nums[i] = nums[i], nums[first]
        # n = len(nums)
        # res = []
        # backtrace(0)
        # return res



# leetcode submit region end(Prohibit modification and deletion)
