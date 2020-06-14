# 给你两个有序整数数组 nums1 和 nums2，请你将 nums2 合并到 nums1 中，使 nums1 成为一个有序数组。 
# 
#  
# 
#  说明: 
# 
#  
#  初始化 nums1 和 nums2 的元素数量分别为 m 和 n 。 
#  你可以假设 nums1 有足够的空间（空间大小大于或等于 m + n）来保存 nums2 中的元素。 
#  
# 
#  
# 
#  示例: 
# 
#  输入:
# nums1 = [1,2,3,0,0,0], m = 3
# nums2 = [2,5,6],       n = 3
# 
# 输出: [1,2,2,3,5,6] 
#  Related Topics 数组 双指针


# leetcode submit region begin(Prohibit modification and deletion)
class Solution:
    def merge(self, nums1: List[int], m: int, nums2: List[int], n: int) -> None:
        """
        Do not return anything, modify nums1 in-place instead.
        """
        # 合并后排序
        # nums1[:] = sorted(nums1[0:m] + nums2)

        # 添加额外空间存储nums1
        # nums1_copy = nums1[:m]
        # nums1[:] = [0]*(m+n)
        # i, j, k = 0, 0, 0
        # while i < m and j < n:
        #     if nums1_copy[i] < nums2[j]:
        #         nums1[k] = nums1_copy[i]
        #         i += 1
        #     else:
        #         nums1[k] = nums2[j]
        #         j += 1
        #     k += 1
        # if i < m:
        #     nums1[i+j:] = nums1_copy[i:]
        # if j < n:
        #     nums1[i+j:] = nums2[j:]
        # 两指针，从后开始
        p1, p2 = m - 1, n - 1
        p = m + n -1
        while p1 >= 0 and p2 >= 0:
            if nums1[p1] > nums2[p2]:
                nums1[p] = nums1[p1]
                p1 -= 1
            else:
                nums1[p] = nums2[p2]
                p2 -= 1
            p -= 1
        nums1[:p2 + 1] = nums2[:p2 + 1]
# leetcode submit region end(Prohibit modification and deletion)
