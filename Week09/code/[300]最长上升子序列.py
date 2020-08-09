# 给定一个无序的整数数组，找到其中最长上升子序列的长度。 
# 
#  示例: 
# 
#  输入: [10,9,2,5,3,7,101,18]
# 输出: 4 
# 解释: 最长的上升子序列是 [2,3,7,101]，它的长度是 4。 
# 
#  说明: 
# 
#  
#  可能会有多种最长上升子序列的组合，你只需要输出对应的长度即可。 
#  你算法的时间复杂度应该为 O(n2) 。 
#  
# 
#  进阶: 你能将算法的时间复杂度降低到 O(n log n) 吗? 
#  Related Topics 二分查找 动态规划


# leetcode submit region begin(Prohibit modification and deletion)
class Solution:
    def lengthOfLIS(self, nums: List[int]) -> int:
        # DP
        # if not nums: return 0
        # N = len(nums)
        # dp = [1] * N
        # for i in range(1, N):
        #     for j in range(i):
        #         if nums[j] < nums[i]:
        #             dp[i] = max(dp[i], dp[j] + 1)
        # return max(dp)

        # 贪心+二分
        N = len(nums)
        if N < 2: return N
        tails = [nums[0]]
        for i in range(1, N):
            if nums[i] > tails[-1]:
                tails.append(nums[i])
            else:
                left, right = 0, len(tails) - 1
                while left < right:
                    mid = (left + right) >> 1
                    if tails[mid] < nums[i]:
                        left = mid + 1
                    else:
                        right = mid
                tails[left] = nums[i]
        return len(tails)


# leetcode submit region end(Prohibit modification and deletion)
