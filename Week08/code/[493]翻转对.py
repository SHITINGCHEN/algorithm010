# 给定一个数组 nums ，如果 i < j 且 nums[i] > 2*nums[j] 我们就将 (i, j) 称作一个重要翻转对。 
# 
#  你需要返回给定数组中的重要翻转对的数量。 
# 
#  示例 1: 
# 
#  
# 输入: [1,3,2,3,1]
# 输出: 2
#  
# 
#  示例 2: 
# 
#  
# 输入: [2,4,3,5,1]
# 输出: 3
#  
# 
#  注意: 
# 
#  
#  给定数组的长度不会超过50000。 
#  输入数组中的所有数字都在32位整数的表示范围内。 
#  
#  Related Topics 排序 树状数组 线段树 二分查找 分治算法


# leetcode submit region begin(Prohibit modification and deletion)
class Solution:
    def reversePairs(self, nums: List[int]) -> int:
        if not nums: return 0
        return self.mergeSort(nums, 0, len(nums) - 1)

    def mergeSort(self, nums, left, right):
        if left >= right: return 0
        mid = (left + right) >> 1
        count = self.mergeSort(nums, left, mid) + self.mergeSort(nums, mid + 1, right)
        cache = [0] * (right - left + 1)
        i, t, c = left, left, 0
        for j in range(mid + 1, right + 1):
            while i <= mid  and nums[i] <= 2 * nums[j]:
                i += 1
            while t <= mid and nums[t] < nums[j]:
                cache[c] = nums[t]
                c += 1; t += 1
            cache[c] = nums[j]
            count += mid - i + 1
            c += 1

        while t <= mid:
            cache[c] = nums[t]
            c += 1; t += 1
        nums[left : right + 1] = cache[:]
        return count

# leetcode submit region end(Prohibit modification and deletion)
