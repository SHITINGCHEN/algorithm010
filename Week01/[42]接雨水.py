# 给定 n 个非负整数表示每个宽度为 1 的柱子的高度图，计算按此排列的柱子，下雨之后能接多少雨水。 
# 
#  
# 
#  上面是由数组 [0,1,0,2,1,0,1,3,2,1,2,1] 表示的高度图，在这种情况下，可以接 6 个单位的雨水（蓝色部分表示雨水）。 感谢 Mar
# cos 贡献此图。 
# 
#  示例: 
# 
#  输入: [0,1,0,2,1,0,1,3,2,1,2,1]
# 输出: 6 
#  Related Topics 栈 数组 双指针


# leetcode submit region begin(Prohibit modification and deletion)
class Solution:
    def trap(self, height: List[int]) -> int:
        # 思路一，对每一列进行求解
        # 找出在当前列(cur)左边最高的柱子(left)和右边最高的柱子(right)
        # 两种情况：
        # 1. cur的高度小于min(left, right): 那么cur可以存的雨水等于min(left, right) - cur_height
        # 2. cur大于等于min(left, right): 存不到雨水
        # ans = 0
        # for cur in range(1, len(height)-1):
        #     max_left, max_right = max(height[0:cur]), max(height[cur + 1:])
        #     min_height = min(max_left, max_right)
        #     if min_height > height[cur]:
        #         ans += min_height - height[cur]
        # return ans
        # 思路二，栈
        ans = 0
        stack = [0]
        for cur in range(1, len(height)):
            if height[cur] <= height[stack[-1]]:
                stack.append(cur)
            else:
                while stack and height[cur] > height[stack[-1]]:
                    h = height[stack.pop()]
                    if not stack:
                        break
                    width = cur - stack[-1] - 1
                    min_height = min(height[stack[-1]], height[cur])
                    ans += width * (min_height - h)
                stack.append(cur)
        return ans


# leetcode submit region end(Prohibit modification and deletion)
