# 给定一个非负整数数组，你最初位于数组的第一个位置。 
# 
#  数组中的每个元素代表你在该位置可以跳跃的最大长度。 
# 
#  判断你是否能够到达最后一个位置。 
# 
#  示例 1: 
# 
#  输入: [2,3,1,1,4]
# 输出: true
# 解释: 我们可以先跳 1 步，从位置 0 到达 位置 1, 然后再从位置 1 跳 3 步到达最后一个位置。
#  
# 
#  示例 2: 
# 
#  输入: [3,2,1,0,4]
# 输出: false
# 解释: 无论怎样，你总会到达索引为 3 的位置。但该位置的最大跳跃长度是 0 ， 所以你永远不可能到达最后一个位置。
#  
#  Related Topics 贪心算法 数组


# leetcode submit region begin(Prohibit modification and deletion)
class Solution:
    def canJump(self, nums: List[int]) -> bool:
        # 斐波拉契的思路，对于每一个位置，把它所能到达的所有位置都记为True
        # if not nums: return False
        # reached = [True] + [False] * (len(nums)-1)
        # for i in range(len(nums)):
        #     if reached[i]:
        #         if i + nums[i] + 1 < len(nums):
        #             reached[(i + 1) : (i + nums[i] + 1)] = [True] * nums[i]
        #         else:
        #             return True

        # return reached[-1]
        # 贪心，从后往前跳，维护endreachable
        if not nums: return False
        endReachable = len(nums) - 1
        for i in range(len(nums) - 1, -1, -1):
            if nums[i] + i >= endReachable:
                endReachable = i
        return endReachable == 0


# leetcode submit region end(Prohibit modification and deletion)
