# 给定一个非负整数数组，你最初位于数组的第一个位置。 
# 
#  数组中的每个元素代表你在该位置可以跳跃的最大长度。 
# 
#  你的目标是使用最少的跳跃次数到达数组的最后一个位置。 
# 
#  示例: 
# 
#  输入: [2,3,1,1,4]
# 输出: 2
# 解释: 跳到最后一个位置的最小跳跃数是 2。
#      从下标为 0 跳到下标为 1 的位置，跳 1 步，然后跳 3 步到达数组的最后一个位置。
#  
# 
#  说明: 
# 
#  假设你总是可以到达数组的最后一个位置。 
#  Related Topics 贪心算法 数组


# leetcode submit region begin(Prohibit modification and deletion)
class Solution:
    def jump(self, nums: List[int]) -> int:
        # 正向遍历，维护每一步能到达的最远的位置
        # 例如，对于数组[2, 3, 1, 2, 4, 2, 3]，初始位置是下标0，从下标0出发，最远可到达下标2。
        # 下标0可到达的位置中，下标1的值是3，从下标1出发可以达到更远的位置，因此第一步到达下标1。
        # 从下标1出发，最远可到达下标4。下标1可到达的位置中，下标4的值是4 ，从下标4出发可以达到更远的位置，因此第二步到达下标4。
        n = len(nums)
        maxPos, end, step = 0, 0, 0
        for i in range(n - 1):
            if maxPos >= i:
                # 当前能到达的最远位置
                maxPos = max(nums[i] + i, maxPos)
                if i == end: # 到达边界则跳
                    end = maxPos
                    step += 1
        return step

# leetcode submit region end(Prohibit modification and deletion)
