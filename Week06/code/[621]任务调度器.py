# 给定一个用字符数组表示的 CPU 需要执行的任务列表。其中包含使用大写的 A - Z 字母表示的26 种不同种类的任务。任务可以以任意顺序执行，并且每个任务
# 都可以在 1 个单位时间内执行完。CPU 在任何一个单位时间内都可以执行一个任务，或者在待命状态。 
# 
#  然而，两个相同种类的任务之间必须有长度为 n 的冷却时间，因此至少有连续 n 个单位时间内 CPU 在执行不同的任务，或者在待命状态。 
# 
#  你需要计算完成所有任务所需要的最短时间。 
# 
#  
# 
#  示例 ： 
# 
#  输入：tasks = ["A","A","A","B","B","B"], n = 2
# 输出：8
# 解释：A -> B -> (待命) -> A -> B -> (待命) -> A -> B.
#      在本示例中，两个相同类型任务之间必须间隔长度为 n = 2 的冷却时间，而执行一个任务只需要一个单位时间，所以中间出现了（待命）状态。 
# 
#  
# 
#  提示： 
# 
#  
#  任务的总个数为 [1, 10000]。 
#  n 的取值范围为 [0, 100]。 
#  
#  Related Topics 贪心算法 队列 数组


# leetcode submit region begin(Prohibit modification and deletion)
class Solution:
    def leastInterval(self, tasks: List[str], n: int) -> int:
        # 排序
        # tasks_count = [0] * 26
        # for task in tasks:
        #     tasks_count[ord(task) - ord('A')] += 1
        # tasks_count.sort()
        # time = 0
        # while tasks_count[25] > 0:
        #     i = 0
        #     while i <= n:
        #         if tasks_count[25] == 0:
        #             break
        #         if i <= 25 and tasks_count[25 - i] > 0:
        #             tasks_count[25 - i] -= 1
        #         time += 1
        #         i += 1
        #     tasks_count.sort()
        # return time

        # DP?
        tasks_count = [0] * 26
        for task in tasks:
            tasks_count[ord(task) - ord('A')] += 1
        tasks_count.sort()
        max_val = tasks_count[25] - 1
        idle_slots = max_val * n
        for i in range(24, -1, -1):
            if tasks_count[i] > 0:
                # 为什么要和max_val比较：若考虑A、B出现次数一样（4次），
                # 先以A进行分配，那么idle_slots中只能放3个B（max_val = 3)，剩下的一个B不能占用idle_slots
                idle_slots -= min(tasks_count[i], max_val)
        return idle_slots + len(tasks) if idle_slots > 0 else len(tasks)

# leetcode submit region end(Prohibit modification and deletion)
