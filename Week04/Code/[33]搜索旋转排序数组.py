# 假设按照升序排序的数组在预先未知的某个点上进行了旋转。 
# 
#  ( 例如，数组 [0,1,2,4,5,6,7] 可能变为 [4,5,6,7,0,1,2] )。 
# 
#  搜索一个给定的目标值，如果数组中存在这个目标值，则返回它的索引，否则返回 -1 。 
# 
#  你可以假设数组中不存在重复的元素。 
# 
#  你的算法时间复杂度必须是 O(log n) 级别。 
# 
#  示例 1: 
# 
#  输入: nums = [4,5,6,7,0,1,2], target = 0
# 输出: 4
#  
# 
#  示例 2: 
# 
#  输入: nums = [4,5,6,7,0,1,2], target = 3
# 输出: -1 
#  Related Topics 数组 二分查找


# leetcode submit region begin(Prohibit modification and deletion)
from typing import List
class Solution:
    def search(self, nums: List[int], target: int) -> int:
        # 二分 1
        # left, right, mid = 0, len(nums) - 1, 0
        # while left <= right:
        #     mid = (left + right) // 2
        #     if nums[mid] == target:
        #         return mid
        #     elif nums[left] < nums[mid] and nums[mid] >= target >= nums[left]:
        #         right = mid - 1
        #     elif nums[left] > nums[mid]:
        #         if target <= nums[mid] or target >= nums[left]:
        #             right = mid - 1
        #         else:
        #             left = mid + 1
        #     else:
        #         left = mid + 1
        # return -1

        # 二分 2
        left, right, mid = 0, len(nums) - 1, 0
        while left < right:
            mid = (left + right) // 2
            if (target >= nums[left]) ^ (target <= nums[mid]) ^ (nums[left] > nums[mid]):
                left = mid + 1
            else:
                right = mid
        return left if left == right and (nums[left] == target) else -1

# if __name__ == '__main__':
#     nums = list(map(int, input()[1:-1].split(',')))
#     target = int(input())
#     s = Solution()
#     result = s.search(nums, target)
#     print(result)
# leetcode submit region end(Prohibit modification and deletion)
