# 给定一个非空的整数数组，返回其中出现频率前 k 高的元素。 
# 
#  
# 
#  示例 1: 
# 
#  输入: nums = [1,1,1,2,2,3], k = 2
# 输出: [1,2]
#  
# 
#  示例 2: 
# 
#  输入: nums = [1], k = 1
# 输出: [1] 
# 
#  
# 
#  提示： 
# 
#  
#  你可以假设给定的 k 总是合理的，且 1 ≤ k ≤ 数组中不相同的元素的个数。 
#  你的算法的时间复杂度必须优于 O(n log n) , n 是数组的大小。 
#  题目数据保证答案唯一，换句话说，数组中前 k 个高频元素的集合是唯一的。 
#  你可以按任意顺序返回答案。 
#  
#  Related Topics 堆 哈希表
# logn：一般是堆、二叉搜索树、排序、二分查找

# leetcode submit region begin(Prohibit modification and deletion)
class ProrityQueue:
    def __init__(self):
        self.arr = []

import heapq as hq
from collections import Counter
class Solution:
    def topKFrequent(self, nums: List[int], k: int) -> List[int]:
        # 维护一个k大小的小顶堆，当新元素大于堆顶时，删除堆顶并将新元素push进去
        # count = Counter(nums)
        # res = []
        # hp = []
        # for num, freq in count.items():
        #     if len(hp) == k:
        #         if hp[0][0] < freq:
        #             hq.heapreplace(hp, (freq, num))
        #     else:
        #         hq.heappush(hp, (freq, num))
        # while hp:
        #     res.append(hq.heappop(hp)[1])
        # return res
        def heapifyup(arr, k):
            # 加入新元素（数组尾部）后
            # 依次向上调整至根
            new_idx, new_val = k-1, arr[k-1]
            father = (new_idx - 1)//2
            while new_idx > 0 and arr[father][1] > new_val[1]:
                arr[new_idx] = arr[father]
                new_idx = father
                father = (new_idx -1) // 2
            arr[new_idx] = new_val


        def heapifydown(arr, k):
            # 新元素比堆顶大，堆顶元素已经替换为新元素了
            # 重新维护堆序（小顶堆）
            root_val, root, son = arr[0], 0, 1
            while son < k:
                if son + 1 < k:
                    if arr[son + 1][1] < arr[son][1]:
                        son = son + 1
                if root_val[1] > arr[son][1]:
                    arr[root] = arr[son]
                    root = son
                    son = root * 2 + 1
                else:
                    break
            arr[root] = root_val


        # 先构造大小为k的小顶堆
        count = list(Counter(nums).items())
        min_heap = []
        for i in range(k):
            min_heap.append(count[i])
            heapifyup(min_heap, i+1)

        for i in range(k, len(count)):
            if count[i][1] > min_heap[0][1]:
                min_heap[0] = count[i]
                heapifydown(min_heap, k)

        return [x[0] for x in min_heap]
# leetcode submit region end(Prohibit modification and deletion)
