# 我们把只包含因子 2、3 和 5 的数称作丑数（Ugly Number）。求按从小到大的顺序的第 n 个丑数。 
# 
#  
# 
#  示例: 
# 
#  输入: n = 10
# 输出: 12
# 解释: 1, 2, 3, 4, 5, 6, 8, 9, 10, 12 是前 10 个丑数。 
# 
#  说明: 
# 
#  
#  1 是丑数。 
#  n 不超过1690。 
#  
# 
#  注意：本题与主站 264 题相同：https://leetcode-cn.com/problems/ugly-number-ii/ 
#  Related Topics 数学


# leetcode submit region begin(Prohibit modification and deletion)
import heapq as hq
class Solution:
    def nthUglyNumber(self, n: int) -> int:
        s = set()
        if n == 1:
            return 1
        hp, primes = [2,3,5], [2,3,5]
        hq.heapify(hp)
        for i in range(1, n):
            num = hq.heappop(hp)
            for x in primes:
                tmp = num * x
                if tmp not in s:
                    hq.heappush(hp, tmp)
                    s.add(tmp)
        return num

# leetcode submit region end(Prohibit modification and deletion)
