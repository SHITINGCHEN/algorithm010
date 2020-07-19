# 一条包含字母 A-Z 的消息通过以下方式进行了编码： 
# 
#  'A' -> 1
# 'B' -> 2
# ...
# 'Z' -> 26
#  
# 
#  给定一个只包含数字的非空字符串，请计算解码方法的总数。 
# 
#  示例 1: 
# 
#  输入: "12"
# 输出: 2
# 解释: 它可以解码为 "AB"（1 2）或者 "L"（12）。
#  
# 
#  示例 2: 
# 
#  输入: "226"
# 输出: 3
# 解释: 它可以解码为 "BZ" (2 26), "VF" (22 6), 或者 "BBF" (2 2 6) 。
#  
#  Related Topics 字符串 动态规划


# leetcode submit region begin(Prohibit modification and deletion)
class Solution:
    def numDecodings(self, s: str) -> int:
        # dp[i]: 以i为结尾有多少种编码方法
        # dp[i] = dp[i - 1] + dp[i - 2]
        if not s: return 0
        if s[0] == '0': return 0
        dp = [0] * (len(s) + 1)
        dp[0] = 1
        dp[1] = 1
        for i in range(2, len(s) + 1):
            if s[i - 1] == '0':
                if s[i - 2] > '2' or s[i - 2] <= '0':
                    return 0
                else:
                    dp[i] = dp[i - 2]
            elif '7' <= s[i - 1] <= '9':
                if s[i - 2] >= '2' or s[i - 2] == '0':
                    dp[i] = dp[i - 1]
                else:
                    dp[i] = dp[i - 1] + dp[i - 2]
            else:
                if s[i - 2] > '2' or s[i - 2] == '0':
                    dp[i] = dp[i - 1]
                else:
                    dp[i] = dp[i - 1] + dp[i - 2]
        # print(dp)
        return dp[len(s)]
# leetcode submit region end(Prohibit modification and deletion)
