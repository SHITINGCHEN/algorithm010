# 编写一个高效的算法来判断 m x n 矩阵中，是否存在一个目标值。该矩阵具有如下特性： 
# 
#  
#  每行中的整数从左到右按升序排列。 
#  每行的第一个整数大于前一行的最后一个整数。 
#  
# 
#  示例 1: 
# 
#  输入:
# matrix = [
#   [1,   3,  5,  7],
#   [10, 11, 16, 20],
#   [23, 30, 34, 50]
# ]
# target = 3
# 输出: true
#  
# 
#  示例 2: 
# 
#  输入:
# matrix = [
#   [1,   3,  5,  7],
#   [10, 11, 16, 20],
#   [23, 30, 34, 50]
# ]
# target = 13
# 输出: false 
#  Related Topics 数组 二分查找


# leetcode submit region begin(Prohibit modification and deletion)
from typing import List
class Solution:
    def searchMatrix(self, matrix: List[List[int]], target: int) -> bool:
        # 一、二维变一维 O(n + logn)
        # arr = [i for sub in matrix for i in sub]
        # if not arr: return False
        # left, right, mid = 0, len(arr) - 1, 0
        # while left <= right:
        #     mid = (left + right) // 2
        #     if arr[mid] == target:
        #         return True
        #     elif arr[mid] > target:
        #         right = mid -1
        #     else:
        #         left = mid + 1
        # return False

        # 二、先确定target在哪个子列表中，再进行二分
        if not matrix or not matrix[0]:
            return False
        left, right, row = 0, len(matrix) - 1, 0
        while left <= right:
            mid = (left + right) // 2
            if matrix[mid][0] <=  target <= matrix[mid][-1]:
                row = mid
                break
            elif matrix[mid][0] > target:
                right = mid - 1
            elif matrix[mid][-1] < target:
                left = mid + 1

        if left >= len(matrix) or right < 0:
            return False

        left, right, mid = 0, len(matrix[0]) - 1, 0
        while left <= right:
            mid = (left + right) // 2
            if matrix[row][mid] == target:
                return True
            elif matrix[row][mid] > target:
                right = mid -1
            else:
                left = mid + 1
        return False

# if __name__ == '__main__':
#     matrix = [[1,3,5,7],[10,11,16,20],[23,30,34,50]]
#     target = 3
#     print('hello')
#     a = Solution()
#     print(a.searchMatrix(matrix, target))
# leetcode submit region end(Prohibit modification and deletion)
