# 在一个 2 x 3 的板上（board）有 5 块砖瓦，用数字 1~5 来表示, 以及一块空缺用 0 来表示. 
# 
#  一次移动定义为选择 0 与一个相邻的数字（上下左右）进行交换. 
# 
#  最终当板 board 的结果是 [[1,2,3],[4,5,0]] 谜板被解开。 
# 
#  给出一个谜板的初始状态，返回最少可以通过多少次移动解开谜板，如果不能解开谜板，则返回 -1 。 
# 
#  示例： 
# 
#  
# 输入：board = [[1,2,3],[4,0,5]]
# 输出：1
# 解释：交换 0 和 5 ，1 步完成
#  
# 
#  
# 输入：board = [[1,2,3],[5,4,0]]
# 输出：-1
# 解释：没有办法完成谜板
#  
# 
#  
# 输入：board = [[4,1,2],[5,0,3]]
# 输出：5
# 解释：
# 最少完成谜板的最少移动次数是 5 ，
# 一种移动路径:
# 尚未移动: [[4,1,2],[5,0,3]]
# 移动 1 次: [[4,1,2],[0,5,3]]
# 移动 2 次: [[0,1,2],[4,5,3]]
# 移动 3 次: [[1,0,2],[4,5,3]]
# 移动 4 次: [[1,2,0],[4,5,3]]
# 移动 5 次: [[1,2,3],[4,5,0]]
#  
# 
#  
# 输入：board = [[3,2,4],[1,5,0]]
# 输出：14
#  
# 
#  提示： 
# 
#  
#  board 是一个如上所述的 2 x 3 的数组. 
#  board[i][j] 是一个 [0, 1, 2, 3, 4, 5] 的排列. 
#  
#  Related Topics 广度优先搜索


# leetcode submit region begin(Prohibit modification and deletion)

# BFS code
# class Solution:
#     def slidingPuzzle(self, board: List[List[int]]) -> int:
#         # 表示空格/0处于第几个位置时，下一步它可移动的位置
#         pos = {0: [1,3],
#                1: [0, 2, 4],
#                2: [1, 5],
#                3: [0, 4],
#                4: [1, 3, 5],
#                5: [2, 4]}
#         start = ''.join(''.join(map(str, i)) for i in board)
#         que, visited = [start], set()
#         step = 0
#         while que:
#             next_level = []
#             for node in que:
#                 visited.add(node)
#                 if node == '123450':
#                     return step
#                 zero_index = node.index('0')
#                 for i in pos[zero_index]:
#                     tmp = list(node)
#                     tmp[i], tmp[zero_index] = tmp[zero_index], tmp[i]
#                     next_state = ''.join(tmp)
#                     if not next_state in visited:
#                         next_level.append(next_state)
#             step += 1
#             que = next_level
#         return -1

# A*解法，以某个数字现在位置和目标位置之间的距离作为估价函数
from collections import namedtuple
import heapq, copy
class Solution:
    def slidingPuzzle(self, board: List[List[int]]) -> int:
        self.scores = [0] * 6
        # 目标位置
        goal_pos = {1:(0,0), 2:(0,1), 3:(0,2), 4:(1,0), 5:(1,1), 0:(1,2)}
        for num in range(6):
            self.scores[num] = [[abs(goal_pos[num][0] - i) + abs(goal_pos[num][1] - j) for j in range (3)] for i in range(2)]

        self.moves = {(0, 0): [(0, 1), (1,0)],
                 (0, 1): [(0, 0), (0, 2), (1, 1)],
                 (0, 2): [(0, 1), (1, 2)],
                 (1, 0): [(0, 0), (1, 1)],
                 (1, 1): [(0, 1), (1, 0), (1, 2)],
                 (1, 2): [(0, 2), (1, 1)]}
        Node = namedtuple('Node', ['heuristic_score', 'steps', 'board'])
        heap = [Node(0, 0, board)]
        visited = []
        while heap:
            node = heapq.heappop(heap)
            if self.get_score(node.board) == 0:
                return node.steps
            elif node.board in visited:
                continue
            else:
                for state in self.get_next_states(node.board):
                    if state not in visited:
                        heapq.heappush(heap, Node(node.steps + 1 + self.get_score(state), node.steps + 1, state))
            visited.append(node.board)
        return -1

    def get_score(self, board):
        return sum([self.scores[board[i][j]][i][j] for i in range(2) for j in range(3)])


    def get_next_states(self, board):
        res = []
        if 0 in board[0]:
            r, c = 0, board[0].index(0)
        else:
            r, c = 1, board[1].index(0)
        for x, y in self.moves[(r, c)]:
            tmp = copy.deepcopy(board)
            tmp[r][c], tmp[x][y] = tmp[x][y], tmp[r][c]
            res.append(tmp)
        return res

# leetcode submit region end(Prohibit modification and deletion)
